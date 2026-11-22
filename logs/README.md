# logs

本目录保存各组实验的汇总文件与训练日志尾部。完整训练日志不入库（单份约 30 MB）。

| 文件 | 内容 |
|---|---|
| `seeds_summary.txt` | 域内 40 epoch，基线 / C1-32，seed 0–2 |
| `c1L_summary.txt` | 域内 40 epoch，C1-64，seed 0–2 |
| `heavy_eval.csv` | 重雾交叉测试：基线 / C1-32 |
| `heavy_L.csv` | 重雾交叉测试：C1-64 |
| `ab_summary.txt` | 分支消融：FFT-only / DWT-only，各 3 个种子 |
| `queue2_summary.txt` | sham 对照、γ 解耦、补充种子 |
| `*-tail.txt` | 代表性训练日志末尾 200 行，用于核对 best_stat 行 |
| `g15_summary.txt` | γ=1.5 扫描，3 个种子 |
| `heavy_all.csv` | 重雾评估全表，含每个权重的绝对路径（base/sham/c1L/g15/fftonly/dwtonly） |
| `probe.csv` | 频率成分探针：原始 / 低通 / 高通三种输入条件下 base 与 C1-64 的评估结果 |
| `subband.csv` | 子带干预：逐一移除 LH/HL/HH 后的评估结果 |
| `genB.csv` | 跨生成器测试：genB 雾模型下的基线 / 频率前端评估 |
| `tta_baseline.csv` | 测试时增强基线（CLAHE / 伽马），五种子评估结果 |
| `probe_mechanism.json` | 机制后验分析：与反锐化掩模的相似度、径向与方向谱增益 |

## 生成方式

- 训练：`python tools/train.py -c <config> -u epoches=40 seed=N checkpoint_freq=40 output_dir=<out>`
- 汇总：从日志中抓取最后一行 `best_stat` 的 `coco_eval_bbox`
- 重雾评估：`python tools/train.py -c configs/rtdetrv2/heavy_*.yml -r <ckpt> --test-only`

## 口径

- 所有数字为 40 epoch 训练结束后的 best AP（COCO AP，IoU 0.50:0.95）
- 同配置同种子重跑，AP 最大偏差 0.011，因此小于 1.5 点的差异不作结论
- 早期 30 epoch 的探索性结果（`grid_summary.txt`）口径不同，不与上表并列
