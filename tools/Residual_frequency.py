"""
integrate model output patches into the original remote sensing size.
"""
import pywt
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
    parser.add_argument('-d', '--dir', required=False, default='F:\\file\pan-sharpening\对比算法\数据\TIF', help='directory of input patches')
    parser.add_argument('-t', '--dst', required=False, default=r'residual_png', help='directory of save path')
    parser.add_argument('-c', '--col', required=False, default=21, type=int, help='how many columns')
    parser.add_argument('-r', '--row', required=False, default=21, type=int, help='how many rows')
    parser.add_argument('--ms_chan', default=4, type=int, help='how many channels of MS')
    parser.add_argument('-p', '--patch_size', default=400, type=int, help='patch size')
    # parser.add_argument('-a', '--attn_block', default=81, type=int, help='the number of patch for show')
    parser.add_argument('-s', '--scale', default=7, type=float, help='the rate of image dowm-sample scale')

    return parser.parse_args()

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

def DWT(img):   # (C, H, W)
    # DWT
    # 定义小波基和分解层数
    wavelet = 'haar'
    # 应用2D DWT
    coeffs = pywt.wavedec2(img, wavelet='haar', level=1)
    # 假设 coeffs 是通过 pywt.dwt2() 函数获得的小波变换结果
    L, H = coeffs
    H = np.concatenate(H, axis=0)
    c_h = H.shape[0]
    c_l = L.shape[0]

    # 各通道相加形成单通道
    L = np.sum(L, axis=0)
    H = np.sum(H, axis=0)

    # # 归一化
    # L = normalize(L)
    # H = normalize(H)

    # 通道归一
    H = H / c_h
    L = L / c_l

    return L, H

def saveAsPng(args, out, model, dst_model_path, png_type):
    # 图像与关键点放缩

    h, w = out.shape[:2]
    size = (int(w // args.scale), int(h // args.scale))
    out_attn = cv2.resize(out, size, interpolation=cv2.INTER_CUBIC)

    # tif转化成png

    if out_attn.ndim == 2:
        img = out_attn.astype(int)
    else:   # 三个维度
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


    # dst_file = dst_model_path + args.data_type + '-' + model + '-' + {png_type}.png'
    # orgin_dst = f'{dst_model_path}/{args.data_type}-{model}-origin.png'
    orgin_dst = f'{dst_model_path}/{args.data_type}-{model}-{png_type}.png'
    img.save(orgin_dst)

def Draw(args, model, src_path, dst_path):
    patch_size = args.patch_size
    # 确定图像尺寸
    y_size = patch_size // 2 * (args.row - 1) + patch_size
    x_size = patch_size // 2 * (args.col - 1) + patch_size
    out = np.zeros(shape=[y_size, x_size, args.ms_chan], dtype=np.float32)
    res_out = np.zeros(shape=[y_size, x_size, args.ms_chan], dtype=np.float32)
    cnt = np.zeros(shape=out.shape, dtype=np.float32)

    g_res_out = np.zeros(shape=[y_size, x_size], dtype=np.float32)
    g_cnt = np.zeros(shape=g_res_out.shape, dtype=np.float32)

    h_res_out = np.zeros(shape=[y_size // 2, x_size // 2], dtype=np.float32)
    l_res_out = np.zeros(shape=[y_size // 2, x_size // 2], dtype=np.float32)
    lh_cnt = np.zeros(shape=[y_size // 2, x_size // 2], dtype=np.float32)

    # print(out.shape)

    # 组成tif块
    i = 0
    y = 0

    for _ in range(args.row):
        x = 0
        for __ in range(args.col):
            ly = y
            ry = y + patch_size
            lx = x
            rx = x + patch_size

            cnt[ly:ry, lx:rx, :] = cnt[ly:ry, lx:rx, :] + 1
            g_cnt[ly:ry, lx:rx] = g_cnt[ly:ry, lx:rx] + 1
            lh_cnt[ly//2:ry//2, lx//2:rx//2] = g_cnt[ly//2:ry//2, lx//2:rx//2] + 1

            gt_img =  f'{args.dir}/{args.data_type}/GT/{i}_mul.tif'
            img = f'{src_path}/{i}_mul_hat.tif'

            img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)   # 模型输出图片
            img = np.array(img, dtype=np.float32)
            gt_img = gdal.Open(gt_img).ReadAsArray().transpose(1, 2, 0)   # 模型输出图片
            gt_img = np.array(gt_img, dtype=np.float32)

            res_img = abs(img - gt_img)

            temp = res_img.transpose(2, 0, 1)   # (C, H, W)
            l_res_img, h_res_img = DWT(temp)
            g_res_img = np.sum(temp, axis=0) / temp.shape[0]

            out[ly:ry, lx:rx, :] = out[ly:ry, lx:rx, :] + img
            res_out[ly:ry, lx:rx, :] = res_out[ly:ry, lx:rx, :] + res_img
            g_res_out[ly:ry, lx:rx] = g_res_out[ly:ry, lx:rx] + g_res_img
            l_res_out[ly//2:ry//2, lx//2:rx//2] = l_res_out[ly//2:ry//2, lx//2:rx//2] + l_res_img
            h_res_out[ly//2:ry//2, lx//2:rx//2] = h_res_out[ly//2:ry//2, lx//2:rx//2] + h_res_img

            i = i + 1
            x = x + patch_size // 2
        y = y + patch_size // 2

    out = out / cnt # (row, col, c)
    res_out = res_out / cnt # (row, col, c)
    g_res_out = g_res_out / g_cnt
    l_res_out = l_res_out / lh_cnt # (row, col)
    h_res_out = h_res_out / lh_cnt # (row, col)

    # 保存图像
    dst_model_path = f'{dst_path}/{model}'
    mmcv.mkdir_or_exist(dst_model_path)
    saveAsPng(args, out, model, dst_model_path, png_type='origin')
    saveAsPng(args, h_res_out, model, dst_model_path, png_type='high')
    saveAsPng(args, l_res_out, model, dst_model_path, png_type='low')
    saveAsPng(args, res_out, model, dst_model_path, png_type='res')
    saveAsPng(args, g_res_out, model, dst_model_path, png_type='gray')

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
    mmcv.mkdir_or_exist(args.dst)
    dst_path = f'{args.dst}/{args.data_type}'
    mmcv.mkdir_or_exist(dst_path)

    file_path = f'{args.dir}/{args.data_type}'
    for idx, model in enumerate(os.listdir(file_path)):
        src_path = None
        if model == "GT":
            # src_path = f'{args.dir}/{model}'
            continue
        else:
            src_path = f'{file_path}/{model}'
        Draw(args, model, src_path, dst_path)

