#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一运行入口：依次执行 S1 → S2 → S3 → S4
=====================================================================
前置条件：pip install pandas numpy scikit-learn lightgbm scipy matplotlib shap
用法：python run_all_supplementary.py
产出目录：D:\Documents\qwen-agent\02jflp9200\default\补充实验产出
=====================================================================
"""
import subprocess, sys, os, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("S1 · 复现实验 + 基线 + 导出预测值", "S1_cells.py"),
    ("S2 · 补充指标与统计检验",           "S2_metrics_and_tests.py"),
    ("S3 · SHAP + CART稳定性 + 时延",      "S3_shap_stability_latency.py"),
    ("S4 · 出版级图件重绘",                "S4_figures.py"),
]

print("=" * 70)
print("补充实验流水线启动")
print("=" * 70)
t0 = time.time()

for i, (desc, fname) in enumerate(STEPS, 1):
    path = os.path.join(SCRIPT_DIR, fname)
    if not os.path.exists(path):
        print(f"\n[{i}/4] ⚠️ 跳过 {desc}：找不到 {path}")
        continue
    print(f"\n{'=' * 70}")
    print(f"[{i}/4] {desc}")
    print(f"{'=' * 70}")
    result = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"\n❌ {fname} 退出码 {result.returncode}，流水线中止。")
        print("   请查看上方报错信息，修正后重新运行本脚本。")
        sys.exit(result.returncode)

elapsed = time.time() - t0
out_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "补充实验产出")
print(f"\n{'=' * 70}")
print(f"全部完成，总耗时 {elapsed:.0f}s（{elapsed/60:.1f} min）")
print(f"产出目录：{out_dir}")
print(f"{'=' * 70}")
print()
print("关键产出文件清单：")
expected = [
    "predictions_test.csv          ← 测试集预测值（表2/3/4复现+新基线）",
    "predictions_train.csv         ← 训练集预测值（含自映射列，勿用于评估）",
    "models_and_scaler.pkl         ← 模型存档（供S3使用）",
    "validation_scaled_features.csv← 验证集特征（供SHAP使用）",
    "S2_补充指标与统计检验.md       ← 分布指标、显著性、事件类、分时效表",
    "S2_overall_metrics.csv        ← 机器可读版指标表",
    "S2_error_series.csv           ← 误差序列（供绘图）",
    "S3_SHAP_稳定性_时延.md         ← SHAP排序、CART阈值、推理时延",
    "S3_shap_values.csv            ← SHAP重要性数值",
    "S3_cart_thresholds.csv        ← 各折分裂阈值明细",
    "S3_latency.csv                ← 分段时延数据",
    "figures/Fig3_tree.tiff        ← 物理状态树（600dpi）",
    "figures/Fig5_pdf.tiff         ← PDF对比+差值子图",
    "figures/Fig6_timeseries.tiff  ← 大风过程时序",
    "figures/Fig7_shap.tiff        ← SHAP beeswarm",
    "figures/FigS_error_dist.tiff  ← 误差分布+Q-Q",
    "figures/FigS_leadtime.tiff    ← 性能随时效变化",
    "figures/FigS_qq_curve.tiff    ← 分位数-分位数曲线",
    "figures/FigS_bin_rmse.tiff    ← 分风速档RMSE带误差棒",
]
for line in expected:
    name = line.split("<")[0].strip()
    fpath = os.path.join(out_dir, name)
    exists = os.path.exists(fpath)
    marker = "✅" if exists else "❌"
    print(f"  {marker} {line}")
print()
print("下一步：打开 S2_补充指标与统计检验.md 和 S3_SHAP_稳定性_时延.md，")
print("将真实数字填入「审稿意见应对策略与回复草稿.md」中的 [方括号] 占位符。")
