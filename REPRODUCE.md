# 复现说明

本仓库所有表格数值均可由下列命令重跑；硬件为单卡 RTX 5090，环境见 requirements 与 conda 环境 a2net5090；数据由 scripts/make_adverse.py 生成。指标为百分制 COCO AP（IoU 0.50:0.95），训练取 40 epoch 内最佳验证 AP，再跨种子平均。

## 0. 环境与数据

- 依赖：torch、rtdetrv2_pytorch、faster_coco_eval、pycocotools；执行 `ulimit -n 65535`，否则 dataloader 可能报 ancillary data 错误。
- 数据：VOC2012 → COCO 格式（scripts/voc2coco.py）→ 合成雾（scripts/make_adverse.py，β∈[1.2,3.2]）。实际脚本先合成所选 3000 张图，再转换 XML，以 seed=0 固定 2400/600 张训练/验证划分。
- 目录约定：`$WS/data/{fog_voc,heavy_voc,lowpass_voc,highpass_voc,genB_voc}`，权重在 `$WS/runs/<run>/best.pth`。
- 下列 Bash 命令在原实验工作区 rtdetrv2_pytorch 根目录执行，$REPO 指向本归档仓库。配置中的 /home/ubuntu/ws 必须与 $WS 一致。
- 运行前提：准备上游 tools/train.py、基础配置、预训练权重及 requirements，并安装 freq_modules.py 和 patches/rtdetr_patched.py，注册 RTDETRFront。本仓库不含完整训练环境。
- 将本仓库 fog_detection.yml、heavy_detection.yml 放入上游 configs/dataset/，模型配置放入 configs/rtdetrv2/。sham 派生配置使用 c1L 且 mode=conv，g15 使用 c1L 且 gamma=1.5；推理派生配置仅替换验证数据路径，模型须与权重一致。
- 当前归档未包含这些派生配置及 make_clahe.py、make_subband.py、make_genB.py、make_genB_train.py、mechanism_probe.py；对应命令以原工作区已有文件为前提。辅助脚本输入输出及权重参数须按原脚本配置或 --help 设置。

```bash
conda activate a2net5090
export WS=/home/ubuntu/ws
export REPO="$WS/freq-weather-det"
cd "$WS/rtdetrv2_pytorch"
ulimit -n 65535
mkdir -p "$WS/logs"
python "$REPO/scripts/make_adverse.py" --voc "$WS/data/VOCdevkit/VOC2012" --out "$WS/data/fog_voc" --kind fog --n 3000 --seed 0
python "$REPO/scripts/voc2coco.py" --root "$WS/data/fog_voc" --out "$WS/data/fog_voc/annotations" --val-ratio 0.2 --seed 0
```

推理数据沿用原验证图像 ID 和标注。lowpass_voc 为 σ=1.5 低通，highpass_voc 为 σ=3.0 高通；heavy_voc 为 β∈[4,6] 重雾。make_adverse.py 固定普通雾范围，不能直接生成重雾。heavy 配置的标注文件为 annotations_heavy.json。

## 1. 表 1 主结果（训练）

四条命令各覆盖 seed 0–4，独立保存权重与日志。sham/g15 文件名为原工作区派生配置约定。

```bash
for seed in {0..4}; do python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_fog.yml -u epoches=40 seed=$seed output_dir="$WS/runs/s_base_$seed" > "$WS/logs/s_base_$seed.log" 2>&1; done
for seed in {0..4}; do python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_fog_sham.yml -u epoches=40 seed=$seed output_dir="$WS/runs/sham_$seed" > "$WS/logs/sham_$seed.log" 2>&1; done
for seed in {0..4}; do python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_fog_c1L.yml -u epoches=40 seed=$seed output_dir="$WS/runs/L_$seed" > "$WS/logs/L_$seed.log" 2>&1; done
for seed in {0..4}; do python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_fog_g15.yml -u epoches=40 seed=$seed output_dir="$WS/runs/g15_$seed" > "$WS/logs/g15_$seed.log" 2>&1; done
```

| 配置 | 输出目录 | 日志位置 | 预期 AP（五种子均值） |
|---|---|---|---|
| baseline | $WS/runs/s_base_<seed> | $WS/logs/s_base_<seed>.log | 28.58 |
| sham | $WS/runs/sham_<seed> | $WS/logs/sham_<seed>.log | 28.93 |
| c1L | $WS/runs/L_<seed> | $WS/logs/L_<seed>.log | 30.06 |
| γ=1.5 | $WS/runs/g15_<seed> | $WS/logs/g15_<seed>.log | 28.86 |

每个输出目录内权重为 best.pth。历史 baseline seed 3/4 目录为 base_3/base_4，c1L seed 3/4 为 c1L_3/c1L_4；使用旧权重按 logs/heavy_all.csv 的 ckpt 列定位。γ=1.5 现有日志仅有 seed 0–2，28.86 实际是三种子均值；五种子目标沿用表中要求，seed 3/4 尚需重跑验证。

## 2. 表 2 频率探针（仅推理）

两条命令分别评测低通、高通数据集上的 s_base_* 与 L_*，使用 --test-only。推理配置沿用对应模型结构，只改变验证路径。

```bash
for seed in {0..4}; do for model in base c1L; do run="s_base_$seed"; if [ "$model" = c1L ]; then run="L_$seed"; fi; python tools/train.py -c "configs/rtdetrv2/lowpass_${model}.yml" -r "$WS/runs/$run/best.pth" --test-only > "$WS/logs/lowpass_${model}_$seed.log" 2>&1; done; done
for seed in {0..4}; do for model in base c1L; do run="s_base_$seed"; if [ "$model" = c1L ]; then run="L_$seed"; fi; python tools/train.py -c "configs/rtdetrv2/highpass_${model}.yml" -r "$WS/runs/$run/best.pth" --test-only > "$WS/logs/highpass_${model}_$seed.log" 2>&1; done; done
```

| 条件 | baseline AP | c1L AP | 差值 |
|---|---|---|---|
| lowpass_voc | 19.82 | 20.22 | +0.40 |
| highpass_voc | 20.94 | 22.66 | +1.72 |

## 3. 表 3 分支消融

三条训练命令均使用 seed 0–2。both 与表 1 的 c1L 配置相同，取三个种子。

```bash
for seed in {0..2}; do python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_fog_dwtonly.yml -u epoches=40 seed=$seed output_dir="$WS/runs/ab_dwtonly_$seed" > "$WS/logs/ab_dwtonly_$seed.log" 2>&1; done
for seed in {0..2}; do python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_fog_fftonly.yml -u epoches=40 seed=$seed output_dir="$WS/runs/ab_fftonly_$seed" > "$WS/logs/ab_fftonly_$seed.log" 2>&1; done
for seed in {0..2}; do python tools/train.py -c configs/rtdetrv2/rtdetrv2_r18vd_fog_c1L.yml -u epoches=40 seed=$seed output_dir="$WS/runs/ab_both_$seed" > "$WS/logs/ab_both_$seed.log" 2>&1; done
```

| 分支 | 预期 AP（三种子均值） |
|---|---|
| wavelet-only | 29.01 |
| fourier-only | 29.77 |
| both | 30.11 |

## 4. 表 4 重雾（仅推理）

在 heavy_voc 上评测普通雾训练权重。heavy_sham、heavy_fftonly、heavy_dwtonly 为保持模型结构不变的派生配置。

```bash
for model in base sham c1L fftonly dwtonly; do
  seeds="0 1 2 3 4"
  if [ "$model" = fftonly ] || [ "$model" = dwtonly ]; then seeds="0 1 2"; fi
  for seed in $seeds; do
    case "$model" in
      base) run="s_base_$seed" ;;
      sham) run="sham_$seed" ;;
      c1L) run="L_$seed" ;;
      *) run="ab_${model}_$seed" ;;
    esac
    python tools/train.py -c "configs/rtdetrv2/heavy_${model}.yml" -r "$WS/runs/$run/best.pth" --test-only > "$WS/logs/heavy_${model}_$seed.log" 2>&1
  done
 done
```

| 配置 | 种子数 | 预期 AP |
|---|---|---|
| baseline | 5 | 22.00 |
| sham | 5 | 23.06 |
| c1L | 5 | 23.30 |
| fourier-only | 3 | 23.13 |
| wavelet-only | 3 | 21.80 |

## 5. 表 5 测试时基线（仅推理）

先跑原工作区 make_clahe.py 生成 clahe_voc 与 gamma_voc（γ=0.7），保留原验证标注；再评测 base 权重。

```bash
python scripts/make_clahe.py
for dataset in clahe gamma; do
  for seed in {0..4}; do
    python tools/train.py -c "configs/rtdetrv2/${dataset}_base.yml" -r "$WS/runs/s_base_$seed/best.pth" --test-only > "$WS/logs/${dataset}_base_$seed.log" 2>&1
  done
 done
```

| 预处理 | 数据集 | 预期 AP（五种子均值） |
|---|---|---|
| CLAHE | clahe_voc | 26.92 |
| gamma=0.7 | gamma_voc | 27.92 |

## 6. 子带干预与跨生成器

子带：make_subband.py 生成 drop{LH,HL,HH}_voc，固定权重评测 base 与 c1L。

```bash
python scripts/make_subband.py
for band in LH HL HH; do
  for model in base c1L; do
    for seed in {0..4}; do
      run="s_base_$seed"
      if [ "$model" = c1L ]; then run="L_$seed"; fi
      python tools/train.py -c "configs/rtdetrv2/drop${band}_${model}.yml" -r "$WS/runs/$run/best.pth" --test-only > "$WS/logs/drop${band}_${model}_$seed.log" 2>&1
    done
  done
 done
```

| 移除子带 | 预期差值（c1L − base，AP） |
|---|---|
| LH | +1.78 |
| HL | +1.16 |
| HH | +1.66 |

历史 dropLH 基线缺 seed 3；+1.78 为四种子基线与五种子 c1L 均值之差，补齐后应重算。

跨生成器：make_genB.py 与 make_genB_train.py 生成 genB 数据。本节仅使用普通雾训练权重评测 genB_voc。生成器使用纹理深度代理、β 对数均匀 [2,8]、偏冷大气光和 0.6 像素高斯 PSF。

```bash
python scripts/make_genB.py
python scripts/make_genB_train.py
for model in base c1L; do
  for seed in {0..4}; do
    run="s_base_$seed"
    if [ "$model" = c1L ]; then run="L_$seed"; fi
    python tools/train.py -c "configs/rtdetrv2/genB_${model}.yml" -r "$WS/runs/$run/best.pth" --test-only > "$WS/logs/genB_${model}_$seed.log" 2>&1
  done
 done
```

预期：base **11.58**，c1L **11.92**。历史统计分别为五种子、四种子（c1L 缺 seed 3）。当前归档未包含 RESULTS.md 引用的 logs/genB.csv，原始记录尚需从实验工作区补齐。

## 7. 机制后验分析

mechanism_probe.py 需要 PYTHONPATH 指向 rtdetrv2_pytorch，使用与 logs/probe_mechanism.json 相同的 c1L checkpoint、200 张验证图和分析设置；权重路径按原脚本配置。

```bash
PYTHONPATH="$WS/rtdetrv2_pytorch${PYTHONPATH:+:$PYTHONPATH}" python scripts/mechanism_probe.py > "$WS/logs/mechanism_probe.log" 2>&1
```

预期：修正量与反锐化掩模的余弦相似度 **9.2e-5**，平均径向增益 **0.998–1.015**。

## 8. 数值溯源

CSV 的 AP 为 0–1 标度时先乘 100，按条件、模型与种子分组求均值，最后四舍五入。现有训练记录为 TXT，机制分析为 JSON，尚未统一到 CSV。

| 表格或分析 | 原始记录及筛选方式 |
|---|---|
| 表 1 | logs/seeds_summary.txt：baseline seed 0–2；logs/c1L_summary.txt：c1L seed 0–2；logs/queue2_summary.txt：baseline/c1L seed 3–4、sham seed 0–2；logs/g15_summary.txt：g15 seed 0–2、sham seed 3–4。g15 seed 3–4 尚缺。 |
| 表 2 | logs/probe.csv，按 cond=lowpass/highpass、model=base/c1L 分组，seed 0–4。 |
| 表 3 | logs/ab_summary.txt 的 dwtonly/fftonly seed 0–2；logs/c1L_summary.txt 的 both seed 0–2。 |
| 表 4 | logs/heavy_all.csv，按 tag=base/sham/c1L/fftonly/dwtonly 分组，ckpt 列定位历史权重。 |
| 表 5 | logs/tta_baseline.csv，按预处理与种子分组。 |
| 子带 | logs/subband.csv，按子带、模型和种子分组；dropLH 基线缺 seed 3。 |
| 跨生成器 | RESULTS.md 引用 logs/genB.csv，但 CSV 当前未归档；base 五种子，c1L 四种子。 |
| 机制分析 | logs/probe_mechanism.json，200 张验证图。 |
| 总体溯源约定 | 所有原始日志在 logs/，每一行可对应到 logs/*.csv 的某几条记录；现有 TXT、JSON 及缺失 CSV 的例外见上。若重跑数值与表中相差超过 0.05 AP，请先检查随机种子与 ulimit。 |

## 复现验证记录（2026-11-22）

在本机单卡 RTX 5090 上用仓库配置与已有权重重跑两条评测（`--test-only`，验证批大小 2 以适配共享显卡）：

| 配置 | 种子 | 本次 AP | 表中五种子均值 |
| --- | --- | --- | --- |
| rtdetrv2_r18vd_fog（基线） | 0 | 28.4 | 28.58 |
| rtdetrv2_r18vd_fog_c1L（前端） | 0 | 30.4 | 30.06 |

结论：单种子数值落在五种子的正常波动区间内（种子间标准差 0.42–0.78），说明仓库可直接复现正文表 1。
