"""
integrate model output patches into the original remote sensing size.
"""
import argparse
import os
import mmcv
import numpy as np
import gdal
import sys
from PIL import Image, ImageDraw
from tif2png import linear
import cv2
sys.path.append('.')
# from methods.utils import save_image

# GF-2:col=22 row=13 attn_block=22  scale=5.5
# GF-1:col=21 row=21 attn_block=81  scale=7

def parse_args():
    parser = argparse.ArgumentParser(description='integrate patches into one big image')
    parser.add_argument('-z', '--data_type', required=False, default=r'GF-1', help='data type')
    parser.add_argument('-d', '--dir', required=False, default='F:\\file\pan-sharpening\对比算法\数据\GF-1-all\GF-1', help='directory of input patches')
    parser.add_argument('-t', '--dst', required=False, default=r'result_png', help='directory of save path')
    parser.add_argument('-m', '--model', required=False, default='BiFormer', help='name of the model')
    parser.add_argument('-c', '--col', required=False, default=21, type=int, help='how many columns')
    parser.add_argument('-r', '--row', required=False, default=21, type=int, help='how many rows')
    parser.add_argument('--ms_chan', default=4, type=int, help='how many channels of MS')
    parser.add_argument('-p', '--patch_size', default=400, type=int, help='patch size')
    parser.add_argument('-a', '--attn_block', default=81, type=int, help='the number of patch for show')
    parser.add_argument('-s', '--scale', default=7, type=float, help='the rate of image dowm-sample scale')

    return parser.parse_args()

def Draw(args, model, src_path):
    patch_size = args.patch_size
    # 确定图像尺寸
    y_size = patch_size // 2 * (args.row - 1) + patch_size
    x_size = patch_size // 2 * (args.col - 1) + patch_size
    out = np.zeros(shape=[y_size, x_size, args.ms_chan], dtype=np.float32)
    cnt = np.zeros(shape=out.shape, dtype=np.float32)
    # print(out.shape)

    # 组成tif块
    i = 0
    y = 0
    x_attn = 0
    y_attn = 0
    img_attn = None
    for _ in range(args.row):
        x = 0
        for __ in range(args.col):
            ly = y
            ry = y + patch_size
            lx = x
            rx = x + patch_size
            cnt[ly:ry, lx:rx, :] = cnt[ly:ry, lx:rx, :] + 1
            # img = f'{args.dir}/{i}_mul_hat.tif'
            img = None
            if model == "GT":
                img = f'{src_path}/{i}_mul.tif'
            else:
                img = f'{src_path}/{i}_mul_hat.tif'
            img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
            img = np.array(img, dtype=np.float32)
            out[ly:ry, lx:rx, :] = out[ly:ry, lx:rx, :] + img
            if i == args.attn_block:
                x_attn = x
                y_attn = y
                img_attn = img  # (row, col, c)

            i = i + 1
            x = x + patch_size // 2
        y = y + patch_size // 2
    out = out / cnt # (row, col, c)

    # 图像与关键点放缩
    h, w = out.shape[:2]
    size = (int(w // args.scale), int(h // args.scale))
    out_attn = cv2.resize(out, size, interpolation=cv2.INTER_CUBIC)
    x_attn, y_attn, patch_size_attn = int(x_attn // args.scale), int(y_attn // args.scale), int(patch_size // args.scale),

    # attn_block覆盖原始图像的右下角
    h_attn, w_attn = out_attn.shape[:2]
    out_attn[h_attn-patch_size:h_attn, w_attn-patch_size:w_attn, :] = img_attn

    # tif转化成png
    out_attn = out_attn.transpose(2, 0, 1)    # (c, row, col)
    out_attn = out_attn.astype(int)
    img = linear(out_attn)
    if img.shape[0] in [4, 8]:  # (row, col, c)
        img = img[(2, 1, 0), :, :]
        img = img.transpose(1, 2, 0)
    elif img.shape[0] is 1:
        _, h, w = img.shape
        img = img.reshape(h, w)

    # 图像格式转换
    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)

    # 绘制图像
    draw = ImageDraw.Draw(img)
    attn_points = [(x_attn, y_attn), (x_attn+patch_size_attn, y_attn+patch_size_attn)]
    h, w = size
    attn2_points = [(h-patch_size, w-patch_size), size]
    draw.rectangle(attn_points, outline='yellow', width=4)
    draw.rectangle(attn2_points, outline='yellow', width=4)

    # 保存图像
    # dst_file = args.dst + '/' + args.data_type + '/' + args.data_type + '-' + model + '.png'
    dst_file = args.dst + '/' + args.data_type + '-' + model + '.png'
    mmcv.mkdir_or_exist(args.dst)
    img.save(dst_file)

    print(f"finish model:{model}")

# GF-1:col=21 row=21 attn_block=81  scale=7
# GF-2:col=22 row=13 attn_block=22  scale=5.5

if __name__ == '__main__':
    # 参数获取
    args = parse_args()
    # if args.data_type == 'GF-1':
    #     args.col, args.row, args.attn_block, args.scale = 21, 21, 102, 7
    # else:
    #     args.col, args.row, args.attn_block, args.scale = 22, 13, 23, 5.5
    for idx, model in enumerate(os.listdir(args.dir)):
        sarc_path = None
        if model == "GT":
            src_path = f'{args.dir}/{model}'
        else:
            src_path = f'{args.dir}/{model}'
        Draw(args, model, src_path)

