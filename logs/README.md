# 日志来源

文件原样复制自 staging/collect/logs。下表根据文件内容、配置路径和最佳 AP 对应实验。汇总文件的具体提取命令在素材中未记录。

| 文件 | 实验与配置 | 产生方式 |
| --- | --- | --- |
| `seeds_summary.txt` | 域内基线与 c1s，seed 0/1/2，分别对应 rtdetrv2_r18vd_fog.yml 与 rtdetrv2_r18vd_fog_c1s.yml | 汇总训练 mAP、AP_S，保留种子和配置路径 |
| `c1L_summary.txt` | 域内 C1L，seed 0/1/2，对应 rtdetrv2_r18vd_fog_c1L.yml | 汇总训练 mAP，seed 2 与 L_2 尾日志 best_stat 一致 |
| `heavy_eval.csv` | 重雾基线与 c1s，seed 0/1/2，对应 heavy_base.yml 与 heavy_c1s.yml | 汇总训练权重的重雾评估 AP50、AP，权重路径和命令未记录 |
| `heavy_L.csv` | 重雾 C1L，seed 0/1/2，对应 heavy_c1L.yml | 汇总重雾评估 AP50、AP，权重路径和命令未记录 |
| `grid_summary.txt` | 早期雾天试跑，涉及基线、c1、c1s | 提取 best_stat，记录时间、配置路径及最佳 epoch，种子和运行命令未记录 |
| `ab_summary.txt` | FFT-only 消融，seed 0/1，对应 rtdetrv2_r18vd_fog_fftonly.yml | 汇总两次运行的 AP |
| `s_base_0_tail.txt` | 域内基线 seed 0，对应 rtdetrv2_r18vd_fog.yml | 完整训练输出末尾 200 行，best_stat 与 seeds_summary 对应 |
| `s_c1s_2_tail.txt` | 域内 c1s seed 2，对应 rtdetrv2_r18vd_fog_c1s.yml | 完整训练输出末尾 200 行，best_stat 与 seeds_summary 对应 |
| `L_2_tail.txt` | 域内 C1L seed 2，对应 rtdetrv2_r18vd_fog_c1L.yml | 完整训练输出末尾 200 行，best_stat 与 c1L_summary 对应 |

原始完整训练日志没有入库，只保留末尾 200 行。三个尾日志均已核对为 200 行，汇总 TXT 和 CSV 保留素材提供的完整文件。

配置位于 configs/，日志原始路径沿用上游 RT-DETRv2 布局。c1s 配置实际 front_hidden=32，结果表沿用任务指定 C1-16 名称并注明对应关系，C1L 为宽度 64。
