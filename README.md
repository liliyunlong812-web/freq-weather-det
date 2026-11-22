# freq-weather-det

频率前端 + RT-DETRv2-S，用于恶劣天气目标检测。实验结果见 [RESULTS.md](RESULTS.md)。

## 结构

```
freq_modules.py             前端模块（DWT + FFT，含 mode="conv" 的 sham 对照）
patches/rtdetr_patched.py   RTDETRFront：把前端接在 backbone 之前
configs/                    数据与模型配置（11 个）
scripts/make_adverse.py     合成雾 / 低照数据
scripts/voc2coco.py         VOC xml → COCO json
logs/                       各组实验汇总与训练日志尾部
```

## 环境

- Python 3.10.20，PyTorch 2.7.1+cu128，单卡 RTX 5090（32 GB）
- 检测器：RT-DETRv2-S（ResNet-18 VD），640 输入，3 层解码器
- 训练：40 epoch，seed 0/1/2（部分实验 0–4），有效 batch 16
- 单 epoch 约 54 s，峰值显存 13.4 GB

## 数据

- 训练 / 验证：VOC2012 合成雾，2400 / 600 图（5696 / 1312 框）
- 跨严重度测试：同 600 图重新合成为重雾（β=4~6），训练时未出现

## 模块

| 编号 | 名称 | 接入点 |
|---|---|---|
| C1 | DualDomainFrontend | 输入图 → backbone 之前 |

前端初始化为恒等（实测偏差 6e-8），增益由检测损失学习。
`mode="conv"` 为 sham 对照：同残差结构、同量级参数、不做任何频率变换。

## 引用与出处

检测骨架：RT-DETR，CVPR 2024，pp. 16965–16974。

## 论文与归档状态

实验代码、配置与日志归档在本仓库；论文正文与图不在仓库内，随投稿包单独维护。数据规模：VOC2012 合成雾 2400 训练 / 600 验证，重雾与第二生成器各 600 张用于跨分布测试。
