import matplotlib.pyplot as plt
import gdal
import numpy as np
import mmcv
import random
import cv2

model_names = []

model_path = ""

dst_path = ""

def sharpen_image(image):
    # 使用拉普拉斯算子进行锐化
    kernel = np.array([[0, -1, 0], [-1, 7, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def transForm(tif_path, channel_num, hot_path, model_name, index):
    tif = gdal.Open(tif_path).ReadAsArray()
    hot = tif[channel_num]


    hot = sharpen_image(hot)  # 应用锐化函数
    plt.imshow(hot, cmap='jet')
    plt.axis('off')  # 控制绘制坐标轴，关闭
    plt.savefig(f"{hot_path}/{model_name}_{index}_{channel_num}_hot.png")
    plt.close()

if __name__ == "__main__":
    # 产生随机数
    index = random.randint(0, 285)
    channel_num = random.randint(0, 3)
    index = 58
    channel_num = 2
    out_dir = dst_path + str(index) + "_" + str(channel_num) + "/"
    mmcv.mkdir_or_exist((out_dir))
    for item in model_names:
        tif_path = f'{model_path}{item}/test_out0/iter_12000/{str(index)}_mul_hat.tif'
        transForm(tif_path, channel_num, out_dir, item, index)
        print(f'channel_num:{channel_num}     index:{index}     {item} is finish')