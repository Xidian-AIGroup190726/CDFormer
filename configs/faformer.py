# ---> GENERAL CONFIG <---
# 模型配置参数文件
from typing import Dict

name = 'FAFormer'
description = 'test on GF-1 FAFormer dataset'

model_type = 'FAFormer'
work_dir = f'data/GF-1/model_out/{name}'
log_dir = f'logs/{model_type.lower()}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

only_test = True
# 加载模型
# checkpoint = f'data/GF-1/pretrain.pth'
pretrained = f'data/pretrain.pth'

# ---> DATASET CONFIG <---
ms_chans = 4 # ms文件通道数量
bit_depth = 10

aug_dict: Dict[str, float] = {'lr_flip': 0.5, 'ud_flip': 0.5}

train_set_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/GF-1/dataset/test_low_res'],
        bit_depth=bit_depth),
    num_workers=2,
    batch_size=1,
    shuffle=True)

# 无监督测试
test_set0_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/GF-1/dataset/test_full_res'],
        bit_depth=bit_depth),
    num_workers=2, # 4
    batch_size=1, # 64
    shuffle=False)

# 有监督测试
test_set1_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/GF-1/dataset/test_low_res'],
        bit_depth=bit_depth),
    num_workers=2,
    batch_size=1,
    shuffle=False)


cuda = True  # 使用gpu
# 加载预训练模型，模型已迭代至30000轮
max_iter = 100000
save_freq = 1000 # 1500 保存模型的轮次频率
test_freq = 30000 # 30000 # 保存模型评估指标
eval_freq = 500 # 是否需要评估模型 先不进行full的评估

# norm_input = True # False 仅修改test的值,train需要在ps_dataset进行修改

# ---> SPECIFIC CONFIG <---
sched_cfg = dict(step_size=5000, gamma=0.95) # 15000 0.9

loss_cfg = dict(
    QNR_loss=dict(w=0.0), # 0.1
    spectral_rec_loss=dict(type='l1', w=0.000), # 0.0
    spatial_rec_loss=dict(type='l1', w=0.000), # 0.0
    rec_loss=dict(type='l1', w=1.0))

optim_cfg = dict(
    core_module=dict(type='AdamW', lr=0.0002))# 0.0002

model_cfg = dict(
    n_feats = 16,   # 注意力机制与卷积使用的通道
    core_module=dict(
        norm_input = False,
        bit_depth=bit_depth,
        norm_type='IN',
        # cross-attn数=transformer层数/2 ms卷积层数,每层卷积的结果分别作为Q pan卷积层数,每层卷积的结果分别作为Q ResBlock数
        n_blocks=(1,1,2), # cross-attention数; ms-attn数; pan-attn数 (1,1,2)
        inn_iters = (2, 1)),

    to_pan_mode='avg')


