# -
用来存储我写论文过程中的代码数据集

## 模块说明

freq_modules.py 提供恶劣天气检测的频率模块。

- C1 双域频率解耦，使用 Haar DWT 子带门控与 FFT 幅度/相位分离。
- C2 根据浅层检测特征生成空间门控，使用主干首阶段的 stride-8 特征。
- C3 提供相位一致性损失与离线 logit 蒸馏损失。

前端接在输入图像与 RT-DETR 主干之间，RTDETRFront 补丁接入 C1。

## 实验结果见 RESULTS.md

实验结果见 [RESULTS.md](RESULTS.md)，日志来源见 [logs/README.md](logs/README.md)。

运行环境为 Python 3.10.20、PyTorch 2.7.1+cu128。模型采用 RT-DETRv2-S R18、640 输入、40 epoch、seed 0/1/2。

## 代码结构（freq_modules.py / patches / configs / scripts / logs）

- `freq_modules.py` 保存频率模块实验版本。
- `patches/rtdetr_patched.py` 保存 RT-DETR 集成补丁。
- `configs/` 保存雾天、重雾及消融配置。
- `scripts/make_adverse.py` 生成恶劣天气图像。
- `scripts/voc2coco.py` 将 VOC 标注转换为 COCO 格式。
- `logs/` 保存汇总、评估 CSV、训练尾日志及来源说明。

运行需使用上游 RT-DETRv2 工程布局，配置 include 依赖上游 runtime、dataloader、optimizer 和模型定义。数据集配置、模型配置及补丁需放入上游对应目录，数据路径按实际位置调整。补丁通过同包相对导入使用 freq_modules.py。
