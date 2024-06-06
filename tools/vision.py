"""
This is a tool to get a quick visual result.
For official visual comparison, please use professional softwares like ENVI, ArcMap, etc.
"""
import gdal
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import os


tif_path = ''

def parse_args():
    parser = argparse.ArgumentParser(description='convert .TIF to visible .png')
    parser.add_argument('-s', '--src_file', default=tif_path, help='source file name')
    parser.add_argument('-i', '--index', type=int, default=81, help='The index of the tif picture')
    parser.add_argument('-w', '--width', default=400, help='')
    parser.add_argument('-a', '--height', default=400, help='')
    return parser.parse_args()

def linear(data):
    img_new = np.zeros(data.shape)
    sum_ = data.shape[1] * data.shape[2]
    for i in range(0, data.shape[0]):
        num = np.zeros(5000)
        prob = np.zeros(5000)
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[2]):
                num[data[i, j, k]] = num[data[i, j, k]] + 1
        for tmp in range(0, 5000):
            prob[tmp] = num[tmp] / sum_
        min_val = 0
        max_val = 0
        min_prob = 0.0
        max_prob = 0.0
        while min_val < 5000 and min_prob < 0.2:
            min_prob += prob[min_val]
            min_val += 1
        while True:
            max_prob += prob[max_val]
            max_val += 1
            if max_val >= 5000 or max_prob >= 0.98:
                break
        for m in range(0, data.shape[1]):
            for n in range(0, data.shape[2]):
                if data[i, m, n] > max_val:
                    img_new[i, m, n] = 255
                elif data[i, m, n] < min_val:
                    img_new[i, m, n] = 0
                else:
                    img_new[i, m, n] = (data[i, m, n] - min_val) / (max_val - min_val) * 255
    return img_new

def transform(src_file):
    img = np.array(gdal.Open(src_file).ReadAsArray())

    if img.ndim is 2:
        h, w = img.shape
        img = img.reshape(1, h, w)
    img = linear(img)

    if img.shape[0] in [4, 8]:
        img = img[(2, 1, 0), :, :]
        img = img.transpose(1, 2, 0)
    elif img.shape[0] is 1:
        _, h, w = img.shape
        img = img.reshape(h, w)

    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    return img

if __name__ == '__main__':
    args = parse_args()

    # 计算整体布局的尺寸（二行五列）
    num_rows = 2
    num_cols = 5
    layout_width = args.width * num_cols
    layout_height = args.height * num_rows
    font_size = 100
    font = ImageFont.load_default()  # 使用默认字体

    # 创建一个白色背景的空白图像
    canvas = Image.new('RGB', (layout_width, layout_height), 'yellow')

    # 创建一个用于绘制的ImageDraw对象
    draw = ImageDraw.Draw(canvas)

    # 遍历文件夹下所有文件夹
    for idx, model in enumerate(os.listdir(args.src_file)):
        # 获得图片路径
        tif_path = None
        if model == "GT":
            tif_path = f'{args.src_file}/{model}/{args.index}_mul.tif'
        else:
            tif_path = f'{args.src_file}/{model}/{args.index}_mul_hat.tif'
        png = transform(tif_path)

        row = idx // num_cols
        col = idx % num_cols

        # 计算子图的左上角坐标
        position = (col * args.width, row * args.height)

        # 打开图片并粘贴到画布上
        canvas.paste(png, position)

        # 绘制文本（图片名字）
        text_width, text_height = draw.textsize(model, font=font)
        # 确保文本在图片内部，这里简单地放在左上角，根据需要调整位置
        text_position = (position[0] + 5, position[1] + 5)
        draw.text(text_position, model, font=font, fill='yellow')

        print(f'finish model:{model}')

    # 显示图像
    canvas.show()
    # 如果需要保存图像，可以使用以下代码
    canvas.save('output_image_grid.jpg')