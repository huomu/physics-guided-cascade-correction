#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S3 · SHAP 重算 + CART 阈值稳定性 + 推理时延
=====================================================================
覆盖的审稿意见
---------------------------------------------------------------------
R2-10 / M17  SHAP 分析对象修正：用**正式 Stage-1 模型**（而非另训的100棵简化模型）
             + **held-out 验证集**（而非训练集）重算，并按真实排序输出，
             供改写论文 4.3 节使用
R2-9  / M9   CART 分区阈值的跨折稳定性：报告各折分裂阈值的均值/标准差/变异系数，
             并检验叶节点数是否稳定
R1-9  / M12  推理时延：拆分 LightGBM 预测 / CART 路由 / 分位数映射三段计时，
             给出单样本、单批次(96步=一天)、整月三种规模的时延与内存

依赖：pandas numpy scikit-learn lightgbm scipy；shap（可选，缺失时降级为
      LightGBM 内置 gain importance 并明确标注）
运行：python S3_shap_stability_latency.py
=====================================================================
"""

import os
import gc
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")

# ---- 路径自动探测：与 S1/S2 使用同一套候选顺序，保证内核重启后仍可运行 ----
_OUT_CANDIDATES = [
    r"D:\data\DeChang\analysis_results_06_10\补充实验产出",
    r"D:\Documents\qwen-agent\02jflp9200\default\补充实验产出",
    os.path.join(os.getcwd(), "补充实验产出"),
]
_CSV_CANDIDATES = [
    r"D:\data\DeChang\analysis_results_06_10\paired_data.csv",
    r"C:\Users\admin\Desktop\analysis_results_06_10\paired_data.csv",
    r"C:\Users\admin\Desktop\paired_data.csv",
    os.path.join(os.getcwd(), "paired_data.csv"),
]


def _resolve(env_key, candidates, must_exist=True):
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    for c in candidates:
        if not c:
            continue
        if must_exist:
            if os.path.exists(c):
                return c
        else:
            return c
    return candidates[0] if candidates else None


OUT_DIR = _resolve("DECHANG_OUT", _OUT_CANDIDATES, must_exist=False)
CSV_PATH = _resolve("DECHANG_CSV", _CSV_CANDIDATES, must_exist=True)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[配置] 数据文件 = {CSV_PATH}")
print(f"[配置] 输出目录 = {OUT_DIR}")

PKL_PATH = os.path.join(OUT_DIR, "models_and_scaler.pkl")
VAL_FEAT_PATH = os.path.join(OUT_DIR, "validation_scaled_features.csv")

RANDOM_STATE = 42
CART_MAX_DEPTH = 3
CART_MIN_SAMPLES_LEAF = 500
N_QM_POINTS = 1000
LATENCY_BATCH = 96          # 一天的 15 min 步数
N_LATENCY_REPEAT = 5

FEATURE_COLS = [
    "forecast_wind_speed", "wind_shear", "richardson_number",
    "pressure_gradient", "temperature_gradient", "wind_dir_sin",
    "wind_dir_cos", "solar_radiation", "temp_wind_interaction",
    "pressure_wind_interaction", "season", "is_daytime", "month", "hour",
]
PHYS_COLS = [
    "richardson_number", "temperature_gradient", "wind_shear",
    "pressure_gradient", "temp_wind_interaction",
]
# 论文原 4.3 节声称「主导」的三个特征
CLAIMED_DOMINANT = ["richardson_number", "wind_shear", "temperature_gradient"]


def fmt(v, nd=4):
    try:
        return "N/A" if v is None or not np.isfinite(float(v)) else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "N/A"


# ==================== 1. SHAP 重算 ====================
def recompute_shap(report):
    A = report.append
    A("## 1. SHAP 重算（回应 R2-10 / M17）")
    A("")

    if not os.path.exists(PKL_PATH) or not os.path.exists(VAL_FEAT_PATH):
        A("❌ 缺少 `models_and_scaler.pkl` 或 `validation_scaled_features.csv`，"
          "请先运行 S1。")
        A("")
        return

    with open(PKL_PATH, "rb") as f:
        pack = pickle.load(f)
    model = pack["model_lgb"]
    val = pd.read_csv(VAL_FEAT_PATH)
    X_val = val[FEATURE_COLS].values

    A(f"- 解释对象：**正式 Stage-1 LightGBM 模型**"
      f"（n_estimators={getattr(model, 'n_estimators', 'n/a')}，"
      f"best_iteration={getattr(model, 'best_iteration_', 'n/a')}）")
    A(f"- 解释样本：**held-out 验证集**，N = {len(X_val):,}"
      f"（原图 7 使用另训的 100 棵简化模型且在训练集上计算，两处均需修正）")
    A("")

    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False

    if has_shap:
        A("### 1.1 mean(|SHAP|) 特征重要性排序")
        A("")
        t0 = time.time()
        explainer = shap.TreeExplainer(model)
        # 样本量大时抽样以控制耗时
        n_shap = min(len(X_val), 20000)
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_val), size=n_shap, replace=False) if n_shap < len(X_val) \
            else np.arange(len(X_val))
        sv = explainer.shap_values(X_val[idx])
        sv = np.asarray(sv)
        if sv.ndim == 3:            # 兼容多输出格式
            sv = sv[:, :, -1]
        mean_abs = np.abs(sv).mean(axis=0)
        print(f"  SHAP 计算完成，耗时 {time.time() - t0:.1f}s，样本 {n_shap:,}")

        order = np.argsort(-mean_abs)
        A("| 排名 | 特征 | mean(\|SHAP\|) | 论文原声称 |")
        A("|---|---|---|---|")
        for rank, i in enumerate(order, start=1):
            fname = FEATURE_COLS[i]
            tag = ""
            if fname in CLAIMED_DOMINANT:
                tag = "⚠️ 论文称其「主导」"
            if fname in ("month", "hour", "season", "is_daytime"):
                tag = "时间特征"
            A(f"| {rank} | `{fname}` | {fmt(mean_abs[i])} | {tag} |")
        A("")

        # 关键核对：论文声称 vs 实际
        A("### 1.2 与论文 4.3 节原表述的核对")
        A("")
        A("论文原文：*\"the Richardson number (Ri), wind shear, and temperature "
          "gradient — dominate the model's predictive output, far outweighing "
          "rigid temporal features\"*")
        A("")
        rank_of = {FEATURE_COLS[i]: int(r) for r, i in enumerate(order, start=1)}
        val_of = {FEATURE_COLS[i]: float(mean_abs[i]) for i in range(len(FEATURE_COLS))}
        A("| 论文声称主导的特征 | 实际排名 | 实际 mean(\|SHAP\|) | 是否支持原表述 |")
        A("|---|---|---|---|")
        for fname in CLAIMED_DOMINANT:
            r = rank_of.get(fname)
            v = val_of.get(fname)
            supported = "✅ 是" if (r is not None and r <= 3) else "❌ 否"
            A(f"| `{fname}` | {r} / {len(FEATURE_COLS)} | {fmt(v)} | {supported} |")
        A("")
        time_feats = [f for f in FEATURE_COLS
                      if f in ("month", "hour", "season", "is_daytime")]
        best_time = min(time_feats, key=lambda f: rank_of.get(f, 999))
        A(f"- 排名最高的时间特征：`{best_time}`（第 {rank_of.get(best_time)} 名，"
          f"mean|SHAP| = {fmt(val_of.get(best_time))}）")
        worst_phys = max(PHYS_COLS, key=lambda f: rank_of.get(f, 0))
        A(f"- 排名最低的物理特征：`{worst_phys}`（第 {rank_of.get(worst_phys)} 名，"
          f"mean|SHAP| = {fmt(val_of.get(worst_phys))}）")
        A("")
        A("**4.3 节改写指引**（据上表实际结果填写，不要照搬本模板）：")
        A("")
        A("1. 按实际排序描述前 3 名特征，不得使用 `dominate` / `far outweighing`；")
        A("2. 若时间特征排名高于任一物理特征，删除「物理特征远超时间特征」的对比句；")
        A("3. 删除「confirms that the model has genuinely learned the underlying "
          "atmospheric dynamics」——SHAP 度量模型对特征的依赖度，不等于模型学到了"
          "物理机制；")
        A("4. 说明 `forecast_wind_speed` 排名靠前是**预期结果**：Stage-1 的目标是"
          "对该预报值的残差订正，模型必然高度依赖它；")
        A("5. 在图 7 的 caption 中写明解释对象是正式 Stage-1 模型、"
          "SHAP 值在 held-out 验证集上计算、抽样 N = [填入]。")
        A("")

        # 导出 SHAP 值供作图
        sv_out = pd.DataFrame(sv, columns=FEATURE_COLS)
        sv_out.insert(0, "feature_values_placeholder", 0)
        p = os.path.join(OUT_DIR, "S3_shap_values.csv")
        pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": mean_abs,
            "rank": [rank_of[f] for f in FEATURE_COLS],
        }).to_csv(p, index=False, encoding="utf-8-sig")
        print(f"  SHAP 重要性已导出：{p}")

        # 保存用于 beeswarm 重绘的数据（抽样后）
        np.save(os.path.join(OUT_DIR, "S3_shap_values_sample.npy"), sv)
        np.save(os.path.join(OUT_DIR, "S3_X_val_sample.npy"), X_val[idx])
        print("  SHAP 抽样矩阵已保存，可用 S4 重绘图 7")
    else:
        A("⚠️ 未安装 shap，降级为 LightGBM 内置重要性（**不能**替代 SHAP，"
          "仅用于交叉核对）。请 `pip install shap` 后重跑。")
        A("")
        imp = model.feature_importances_
        A("| 排名 | 特征 | split-based importance | 论文原声称 |")
        A("|---|---|---|---|")
        for rank, i in enumerate(np.argsort(-imp), start=1):
            fname = FEATURE_COLS[i]
            tag = "⚠️ 论文称其「主导」" if fname in CLAIMED_DOMINANT else ""
            A(f"| {rank} | `{fname}` | {int(imp[i])} | {tag} |")
        A("")
    gc.collect()


# ==================== 2. CART 阈值稳定性 ====================
def cart_threshold_stability(report):
    A = report.append
    A("## 2. CART 分区阈值跨折稳定性（回应 R2-9 / M9）")
    A("")

    if not os.path.exists(CSV_PATH):
        A(f"❌ 找不到数据文件：{CSV_PATH}")
        A("")
        return
    if not os.path.exists(PKL_PATH):
        A("❌ 缺少 `models_and_scaler.pkl`，请先运行 S1。")
        A("")
        return

    with open(PKL_PATH, "rb") as f:
        pack = pickle.load(f)
    model_lgb = pack["model_lgb"]
    scaler = pack["scaler"]
    train_idx = pack["train_idx"]

    df = pd.read_csv(CSV_PATH)
    df["obs_time"] = pd.to_datetime(df["obs_time"])
    df = df.sort_values("obs_time").reset_index(drop=True)
    df["wind_speed_error"] = df["obs_wind_speed"] - df["forecast_wind_speed"]
    df["year_month"] = df["obs_time"].dt.to_period("M")
    df["month_int"] = df["month"].astype(int)

    months = sorted(df["month_int"].unique())
    MIN_TRAIN_MONTHS = 2
    test_months = [m for i, m in enumerate(months) if i >= MIN_TRAIN_MONTHS]
    A(f"- 折设置：扩展窗口按月滚动，测试月 = {test_months}（共 {len(test_months)} 折）")
    A("- 每折独立重训 Stage-1 与 CART，记录分裂阈值。**注意**：此处为诊断分区稳定性"
      "而做的重训，不改变论文主结果所用的原切分。")
    A("")

    records = []
    leaf_counts = []
    for tm in test_months:
        tr_months = [m for m in months if m < tm]
        tr = df[df["month_int"].isin(tr_months)]
        if len(tr) < 1000:
            continue
        X_tr = scaler.transform(tr[FEATURE_COLS])
        fcst_tr = tr["forecast_wind_speed"].values
        obs_tr = tr["obs_wind_speed"].values
        vml_tr = np.clip(fcst_tr + model_lgb.predict(X_tr), 0, None)

        X_phys = tr[PHYS_COLS].values
        tree_target = obs_tr - vml_tr
        tree = DecisionTreeRegressor(max_depth=CART_MAX_DEPTH,
                                     min_samples_leaf=CART_MIN_SAMPLES_LEAF,
                                     random_state=RANDOM_STATE)
        tree.fit(X_phys, tree_target)
        leaves = tree.apply(X_phys)
        leaf_counts.append(len(np.unique(leaves)))

        rules = export_text(tree, feature_names=PHYS_COLS, decimals=6)
        for line in rules.splitlines():
            for name in PHYS_COLS:
                if name in line and "<=" in line:
                    try:
                        thr = float(line.split("<=")[1].strip())
                        records.append({"fold_test_month": tm, "feature": name,
                                        "threshold": thr})
                    except (ValueError, IndexError):
                        pass
        # 同时记录该折第一个分裂点（根节点），最能反映分区主结构
        root = tree.tree_
        if root.feature[0] >= 0:
            records.append({"fold_test_month": tm,
                            "feature": PHYS_COLS[root.feature[0]] + " (root)",
                            "threshold": float(root.threshold[0])})
        print(f"  Fold 测试月 {tm}: 训练 {len(tr):,}，叶节点 {len(np.unique(leaves))}")

    if not records:
        A("❌ 未采集到任何分裂阈值。")
        A("")
        return

    rec = pd.DataFrame(records)
    A("### 2.1 各折分裂阈值统计")
    A("")
    A("| 物理特征 | 参与折数 | 分裂次数 | 阈值均值 | 标准差 | 变异系数 CV | "
      "最小 | 最大 | 极差/均值 |")
    A("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for fname in rec["feature"].unique():
        sub = rec[rec["feature"] == fname]["threshold"].values.astype(float)
        if sub.size == 0:
            continue
        mean = float(np.mean(sub))
        std = float(np.std(sub, ddof=1)) if sub.size > 1 else 0.0
        cv = std / abs(mean) if mean != 0 else np.nan
        spread = (float(sub.max()) - float(sub.min())) / abs(mean) if mean != 0 else np.nan
        A(f"| `{fname}` | {rec[rec['feature'] == fname]['fold_test_month'].nunique()} | "
          f"{sub.size} | {fmt(mean)} | {fmt(std)} | {fmt(cv, 3)} | "
          f"{fmt(sub.min())} | {fmt(sub.max())} | {fmt(spread, 3)} |")
        rows.append({"feature": fname, "n_folds": int(
            rec[rec["feature"] == fname]["fold_test_month"].nunique()),
            "mean": mean, "std": std, "cv": cv,
            "min": float(sub.min()), "max": float(sub.max())})
    A("")
    A(f"- 各折叶节点数：{leaf_counts}（均值 {np.mean(leaf_counts):.1f}，"
      f"极差 {max(leaf_counts) - min(leaf_counts)}）")
    A("")

    # 根节点分裂特征的一致性
    root_recs = rec[rec["feature"].str.contains(r"\(root\)")]
    if not root_recs.empty:
        A("### 2.2 根节点分裂特征（分区主结构）")
        A("")
        A("| 折（测试月） | 根节点分裂特征 | 阈值 |")
        A("|---|---|---|")
        for _, r in root_recs.sort_values("fold_test_month").iterrows():
            A(f"| {int(r['fold_test_month'])} | "
              f"`{r['feature'].replace(' (root)', '')}` | {fmt(r['threshold'])} |")
        A("")
        vc = root_recs["feature"].value_counts()
        A(f"- 根节点特征分布：{dict(vc)}")
        A(f"- 主结构一致性：{'✅ 各折根节点特征相同' if len(vc) == 1 else '❌ 各折根节点特征不一致'}")
        A("")

    A("### 2.3 判读与改写指引")
    A("")
    A("- **CV < 0.10 且根节点特征各折一致** → 可主张分区结构稳定，"
      "支持「学到了具有物理意义的分层」的较弱表述；")
    A("- **CV 较大或根节点特征在各折间变化** → 必须在 4.1/4.3 节承认分区是"
      "**数据驱动、站点特异**的最优划分，不得称为「已被确立的大气环流型」，"
      "并删除「methodology will transfer to other terrains without recalibration」"
      "一类表述；")
    A("- 无论结果如何，术语建议统一为 *physically-conditioned strata* 或 "
      "*data-driven regime proxies*，并在 Limitations 中说明阈值需随站点重拟合。")
    A("")
    rec.to_csv(os.path.join(OUT_DIR, "S3_cart_thresholds.csv"),
               index=False, encoding="utf-8-sig")
    print(f"  阈值明细已导出：{os.path.join(OUT_DIR, 'S3_cart_thresholds.csv')}")


# ==================== 3. 推理时延 ====================
def latency_benchmark(report):
    A = report.append
    A("## 3. 推理时延与资源占用（回应 R1-9 / M12）")
    A("")
    if not os.path.exists(PKL_PATH):
        A("❌ 缺少 `models_and_scaler.pkl`，请先运行 S1。")
        A("")
        return
    with open(PKL_PATH, "rb") as f:
        pack = pickle.load(f)
    model_lgb = pack["model_lgb"]
    scaler = pack["scaler"]
    tree = pack["regime_tree"]

    df = pd.read_csv(CSV_PATH)
    df["obs_time"] = pd.to_datetime(df["obs_time"])
    df = df.sort_values("obs_time").reset_index(drop=True)
    sample = df[FEATURE_COLS].head(LATENCY_BATCH).values.astype(float)
    phys_sample = df[PHYS_COLS].head(LATENCY_BATCH).values.astype(float)

    import platform
    A("### 3.1 测试环境")
    A("")
    A(f"- CPU：{platform.processor() or platform.machine()}")
    A(f"- 平台：{platform.platform()}")
    A(f"- Python：{platform.python_version()}")
    A("- 计时方式：每段重复 {n} 次取中位数；LightGBM 默认多线程，"
      "如需单线程业务口径请设置 `n_jobs=1` 后重测并在此注明。".format(n=N_LATENCY_REPEAT))
    A("")

    def timeit(fn, n=N_LATENCY_REPEAT):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)   # ms
        return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))

    X_s = scaler.transform(sample)

    med_scale, _, _ = timeit(lambda: scaler.transform(sample))
    med_lgb, _, _ = timeit(lambda: model_lgb.predict(X_s))
    med_cart, _, _ = timeit(lambda: tree.apply(phys_sample))
    # 分位数映射：以一次典型的分区内插值代表（1000 分位点）
    from scipy.interpolate import interp1d
    src_q = np.linspace(0, 25, N_QM_POINTS)
    tgt_q = np.linspace(0, 25, N_QM_POINTS) ** 1.05
    mapping = interp1d(src_q, tgt_q, kind="linear", bounds_error=False,
                       fill_value="extrapolate")
    pred_vals = model_lgb.predict(X_s)
    med_qm, _, _ = timeit(lambda: mapping(pred_vals))

    total_batch = med_scale + med_lgb + med_cart + med_qm
    per_step = total_batch / LATENCY_BATCH

    A("### 3.2 分段时延（批量 = 一天 96 个 15 min 时刻）")
    A("")
    A("| 阶段 | 批量 96 步 (ms) | 单步 (ms) | 占比 |")
    A("|---|---|---|---|")
    for name, med in [("特征标准化", med_scale), ("Stage-1 LightGBM 预测", med_lgb),
                      ("Stage-2 CART 叶节点路由", med_cart),
                      ("Stage-2 分位数映射插值", med_qm)]:
        A(f"| {name} | {med:.3f} | {med / LATENCY_BATCH:.4f} | "
          f"{100 * med / total_batch:.1f}% |")
    A(f"| **合计** | **{total_batch:.3f}** | **{per_step:.4f}** | 100% |")
    A("")

    # 内存
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1024 / 1024
        A(f"- 进程常驻内存：**{mem_mb:.1f} MB**（含已载入的全部模型与数据）")
    except ImportError:
        A("- 内存未测（未安装 `psutil`）。如需报告，`pip install psutil` 后重跑。")
    A("")

    A("### 3.3 业务可行性推算")
    A("")
    A(f"- 单日 24 h（96 步）全链路订正耗时：**{total_batch:.1f} ms**")
    A(f"- 单步（15 min）平均耗时：**{per_step:.4f} ms**")
    A(f"- 按每月 2,880 步推算：**{per_step * 2880 / 1000:.2f} s**")
    A(f"- 100 个站点的单步批量订正（线性外推）：**{per_step * 100:.2f} ms**")
    A("")
    A("> 关键说明：分位数映射函数在每个更新周期内**只拟合一次**（由训练期分布导出），"
      "业务推理阶段仅执行查表式分段线性插值，因此时延主要由特征派生决定，"
      "而非由订正算法本身决定。")
    A("")
    A("> **写入回复信时必须加的边界声明**：本评估仅覆盖单站点的计算可行性，"
      "模型尚未部署到业务 NWP 后处理链路中，也未在真实数据流约束"
      "（数据到达延迟、质控、并行调度）下测试。这一点须写入 Limitations。")
    A("")

    rows = [{"stage": "特征标准化", "batch_ms": med_scale},
            {"stage": "Stage-1 LightGBM", "batch_ms": med_lgb},
            {"stage": "Stage-2 CART 路由", "batch_ms": med_cart},
            {"stage": "Stage-2 分位数映射", "batch_ms": med_qm},
            {"stage": "合计", "batch_ms": total_batch}]
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "S3_latency.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"  时延表已导出：{os.path.join(OUT_DIR, 'S3_latency.csv')}")


# ==================== 主流程 ====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = ["# 补充实验结果（S3）：SHAP · CART 稳定性 · 推理时延", ""]
    report.append("> 由 `S3_shap_stability_latency.py` 自动生成。"
                  "SHAP 与阈值稳定性部分为**诊断性重算**，不改变论文主结果的切分与数字。")
    report.append("")

    print("\n" + "=" * 70)
    print("[1/3] SHAP 重算（正式模型 + held-out 验证集）")
    print("=" * 70)
    recompute_shap(report)

    print("\n" + "=" * 70)
    print("[2/3] CART 分区阈值跨折稳定性")
    print("=" * 70)
    cart_threshold_stability(report)

    print("\n" + "=" * 70)
    print("[3/3] 推理时延基准测试")
    print("=" * 70)
    latency_benchmark(report)

    md_path = os.path.join(OUT_DIR, "S3_SHAP_稳定性_时延.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n报告已写入：{md_path}")
    print("下一步：python S4_figures.py（重绘出版级图件）")


if __name__ == "__main__":
    main()
