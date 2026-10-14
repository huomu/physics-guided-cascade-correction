# 补充实验代码说明

## 背景

本目录包含为回应两位审稿人（R1 共 10 条、R2 共 14 条）意见而编写的补充实验代码。

**设计原则**：不改动原实验的任何设定（切分协议、模型参数、数据口径），论文表 2/3/4
的原有数字不变。仅在此基础上**新增**基线模型与补充指标。

## 文件清单

| 文件 | 用途 | 覆盖的审稿意见 |
|---|---|---|
| `S1_run_and_export.py` | 复现原实验 + 4个新基线 + 导出预测值 CSV | R1-6 |
| `S2_metrics_and_tests.py` | 从预测值计算全套补充指标 | R1-8, R2-5, R2-6, R2-7, R2-8, R2-12 |
| `S3_shap_stability_latency.py` | SHAP重算 + CART阈值稳定性 + 推理时延 | R1-9, R2-9, R2-10 |
| `S4_figures.py` | 出版级图件重绘（600 dpi TIFF） | R2-14 |
| `run_all_supplementary.py` | 一键顺序执行 S1→S2→S3→S4 | — |

## 运行方式

### 前置依赖

```bash
pip install pandas numpy scikit-learn lightgbm scipy matplotlib shap psutil
```

### 一键运行

```bash
python run_all_supplementary.py
```

或分步执行：

```bash
python S1_run_and_export.py
python S2_metrics_and_tests.py
python S3_shap_stability_latency.py
python S4_figures.py
```

### 数据路径

脚本默认读取：`C:\Users\admin\Desktop\analysis_results_06_10\paired_data.csv`

若数据位置不同，请修改 `S1_run_and_export.py` 顶部的 `CSV_PATH` 变量。

### 产出目录

所有结果输出到：`D:\Documents\qwen-agent\02jflp9200\default\补充实验产出\`

## 各步骤详细说明

### S1：复现 + 基线 + 导出

- **完全复现**原实现的每月内部 70/15/15 交错切分
- Stage-1 LightGBM 原参数（800棵/lr0.03/depth7/leaves31/subsample0.8/colsample0.8）
- Stage-2 CART(max_depth=3, min_samples_leaf=500) + 1000分位点 + extrapolate
- 内置自检：打印与论文表2 Raw行的逐格比对
- 新增基线：F(RF) / G(HistGB) / H(Persistence) / I(Linear bias correction)
- 同时复现表4消融：C(NWP+PC-RQM) / D(LightGBM+Global QM)
- 导出 `predictions_test.csv`（后续 S2/S3 的唯一输入）

### S2：补充指标

从 `predictions_test.csv` 直接计算，**不重训任何模型**：
- 确定性指标 + 方差比 + 偏度/峰度 + Wasserstein + KS + 分位数误差
- 按日分块的 block bootstrap 95% CI + 配对显著性检验
- POD / FAR / TS / HSS（≥10 / ≥12 / ≥15 m/s 三个阈值）
- 分风速段 RMSE（含独立观测时刻数与独立事件天数）
- 分预报时效性能表（0–24h / 24–48h / 48–72h / >72h）
- 误差分布诊断表
- 点精度 ↔ 分布还原权衡汇总表

### S3：SHAP + 稳定性 + 时延

- 用**正式 Stage-1 模型**（非简化版）在**held-out 验证集**（非训练集）上重算 SHAP
- 输出 mean(|SHAP|) 排序表 + 与论文原声称的核对表 + 4.3节改写指引
- 扩展窗口按月滚动重训 CART，记录各折分裂阈值的均值/标准差/CV
- 测量 LightGBM / CART / QM 三段推理时延（批量96步 + 单步 + 月度推算）

### S4：图件重绘

统一规范：600 dpi TIFF + PNG，坐标轴 ≥11pt，刻度 ≥9pt，四模型配色全文一致。
- Fig3 物理状态树（放大标签 + 叶节点统计表）
- Fig5 PDF对比（主图 + 差值子图）
- Fig6 大风过程时序（标注起报时刻与 lead time）
- Fig7 SHAP beeswarm
- FigS 误差分布 + Q-Q / 性能随时效 / 分位数曲线 / 分风速档RMSE带误差棒

## 注意事项

1. `predictions_train.csv` 中的 `E_cascade_selfmap_DO_NOT_EVALUATE` 列是训练集
   自映射结果，方差比人为接近 1，**不可用于任何评估或作图**。
2. CART 的 `min_samples_leaf=500` 与原代码一致；论文 2.5 节写的是 1000，
   需在回复信中说明这一差异。
3. 早停使用验证集（val_idx），不使用测试集，消除 target leakage。
4. 全部脚本均为补充性质，不改变论文原有结果数字。
