import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *

from PIL import Image
import matplotlib.pyplot as plt


def apply_mask(image_path, mask_path, output_path, threshold=128):
    # Load the original image
    image = Image.open(image_path).convert("RGBA")
    # Load the mask image
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

    # Convert images to NumPy arrays
    image_data = np.array(image)
    mask_data = np.array(mask)

    # Ensure the mask is the same size as the image
    if mask_data.shape != image_data.shape[:2]:
        raise ValueError("Mask size does not match image size")
    
    # Set alpha channel to 0 wherever the mask is not white
    # original_alpha = image_data[:, :, 3]
    # image_data[:, :, 3] = np.where(mask_data >= threshold, 255, original_alpha)
    image_data[:, :, 3] = np.where(mask_data >= threshold, 255, 0)

    # Convert back to Image
    result_image = Image.fromarray(image_data, 'RGBA')
    # Save the resulting image
    result_image.save(output_path, format='PNG')

def color_distribution(mask_image_path):
    # 加载图像
    img = Image.open(mask_image_path)
    
    # 确保图像是灰度图
    if img.mode != 'L':
        img = img.convert('L')
    
    # 将图像转换为numpy数组
    data = np.array(img)
    
    # 统计每个颜色值的出现次数
    counts = np.zeros(256, dtype=int)
    for value in range(256):
        counts[value] = np.count_nonzero(data == value)
    
    # 计算累积分布
    cumulative_distribution = np.cumsum(counts)
    total_pixels = data.size
    cumulative_percentage = (cumulative_distribution / total_pixels) * 100
    
    return cumulative_percentage

def plot_distribution_with_derivative(distribution, dis_save_path=None):
    first_derivative = np.diff(distribution)
    second_derivative = np.diff(first_derivative)    
    # 创建图表和两个坐标轴
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # 绘制累积分布
    color = 'tab:blue'
    ax1.set_xlabel('Grayscale Value')
    ax1.set_ylabel('Cumulative Percentage (%)', color=color)
    ax1.plot(distribution, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # 创建第二个坐标轴用于一阶导数
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('1st Derivative', color=color)
    ax2.plot(range(1, len(distribution)), first_derivative, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(("axes", 1.15))  # 将第三个坐标轴稍微偏移
    color = 'tab:green'
    ax3.set_ylabel('Second Derivative', color=color)
    ax3.plot(range(2, len(distribution)), second_derivative, color=color, label='Second Derivative')
    ax3.tick_params(axis='y', labelcolor=color)

    # 添加标题
    plt.title('Cumulative Color Distribution with 1st and 2nd Derivatives')
    
    # 显示图表
    # plt.show()
    if dis_save_path:
        plt.savefig(dis_save_path)

def find_thresholds(cumulative_distribution, epsilon=0.001):
    first_derivative = np.diff(cumulative_distribution)
    # points = []
    ranges = []
    enter_point = 0
    for point in range(1, len(first_derivative)):
        # print(f"first_derivative[point-1] {first_derivative[point-1]} first_derivative[point] {first_derivative[point]}")
        if first_derivative[point-1] > epsilon and first_derivative[point] < epsilon:
            # print(f"point {point}")
            # points.append(point)
            enter_point = point
        if first_derivative[point-1] < epsilon and first_derivative[point] > epsilon:
            ranges.append((enter_point, point))
    ranges.append((enter_point, len(first_derivative)))
    points = [p for r in ranges for p in (r[0], (r[0]+r[1])//2, r[1])]
    # remove duplicate points
    points = list(set(points))
    
    return points

def threshold_inference(im_path, mask_im_path, working_folder_path, result_folder_path):
    im_name = os.path.basename(im_path).split(".")
    im_name = '.'.join(im_name[:-1])
    cumulative_distribution = color_distribution(mask_im_path)
    plot_distribution_with_derivative(cumulative_distribution, dis_save_path=os.path.join(working_folder_path, f"{im_name}_mask_distribution.png"))
    thresholds = find_thresholds(cumulative_distribution, epsilon=0.1)
    # print(f"thresholds {thresholds}")
    # 加几个默认阈值
    thresholds.extend([64, 128])
    for threshold in thresholds:
        apply_mask(im_path, mask_im_path, os.path.join(result_folder_path, f"{im_name}.{threshold}.png"), threshold)
    
def batch_inference(input_folder_path, working_folder_path, model_path, result_folder_path):
    if not os.path.exists(working_folder_path):
        os.makedirs(working_folder_path)
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()
    im_list = glob(input_folder_path+"/*.jpg")+glob(input_folder_path+"/*.JPG")+glob(input_folder_path+"/*.jpeg")+glob(input_folder_path+"/*.JPEG")+glob(input_folder_path+"/*.png")+glob(input_folder_path+"/*.PNG")+glob(input_folder_path+"/*.bmp")+glob(input_folder_path+"/*.BMP")+glob(input_folder_path+"/*.tiff")+glob(input_folder_path+"/*.TIFF")
    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            # print("im_path: ", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            result=net(image)
            result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            # remove extension
            im_name = os.path.basename(im_path).split(".")
            im_name = '.'.join(im_name[:-1])

            # copy input image to working folder and result folder
            os.system(f"cp {im_path} {working_folder_path}")
            os.system(f"cp {im_path} {result_folder_path}")

            mask_path = os.path.join(working_folder_path, im_name + "_mask.png")
            # io.imsave(os.path.join(result_folder_path,im_name+".png"),(result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))
            io.imsave(mask_path, (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))

            # cumulative_distribution = color_distribution(mask_path)
            # plot_distribution_with_derivative(cumulative_distribution, dis_save_path=os.path.join(result_folder_path, f"{im_name}_mask_distribution.png"))
            threshold_inference(im_path, mask_path, working_folder_path, result_folder_path)

if __name__ == "__main__":
    model_path="../saved_models/isnet-general-use.pth"  # the model path
    input_folder_path="./run_input/选图无水印2"  #Your dataset path
    working_folder_path="./run_output/20240714_working_4"  #The folder path that you want to save the working images
    result_folder_path="./run_output/20240714_result_4"  #The folder path that you want to save the results

    batch_inference(input_folder_path, working_folder_path, model_path, result_folder_path)



    # batch_inference(input_folder_path, model_path, result_folder_path)
    # batch_threshold_inference(result_folder_path)

    # cumulative_distribution = color_distribution("./run_output/20240714_result/7.60069_mask.png")
    # plot_distribution_with_derivative(cumulative_distribution, dis_save_path="./run_output/20240714_result/7.60069_mask_distribution.png")

    # thresholds = find_thresholds(cumulative_distribution, epsilon=0.1)
    # print(f"thresholds {thresholds}")
    # for threshold in thresholds:
    #     apply_mask(f"{input_folder_path}/7.60069.jpg", f"{result_folder_path}/7.60069_mask.png", f"{result_folder_path}/7.60069_mask.{threshold}.png", threshold)
