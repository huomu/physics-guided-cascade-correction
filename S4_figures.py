#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S4 · 出版级图件重绘（回应 R2-14 / M19）
=====================================================================
审稿人指出：CART 可视化标签过小、分布图无法读出定量差异、
整体字号/分辨率/线宽/配色不一致。本脚本按统一规范重绘全部图件。

统一规范
---------------------------------------------------------------------
· 输出 600 dpi TIFF（投稿用）+ PNG（预览用）
· 坐标轴标签 ≥10 pt，刻度 ≥9 pt，图例 ≥9 pt，注释 ≥8 pt
· 线宽分层：观测 2.0 / 原始预报 1.5 / Stage-1 1.5 / 级联 2.0
· 单一配色方案，四模型全文一致
· 分布图**必须附差值子图**（cascade − observed），使定量差异可读

产出图件
---------------------------------------------------------------------
Fig3_tree           物理状态树：改为可读的规则图 + 叶节点统计表
Fig5_pdf            PDF 对比（主图 + 差值子图）
Fig6_timeseries     大风过程时序快照（标注起报时刻与 lead time）
Fig7_shap           SHAP beeswarm（用 S3 保存的正式模型结果）
FigS_error_dist     误差分布：直方图 + 正态 Q-Q（回应 R1-8）
FigS_leadtime       性能随预报时效变化（回应 R2-12）
FigS_qq_curve       分位数-分位数曲线（回应 R2-5）
FigS_bin_rmse       分风速档 RMSE 带误差棒（回应 R2-7）

依赖：pandas numpy matplotlib scipy；shap（可选，用于 Fig7）
运行：python S4_figures.py
=====================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats

warnings.filterwarnings("ignore")

# ---- 输出目录：与 S1/S2/S3 使用同一套候选顺序，保证能读到上游产出 ----
_OUT_CANDIDATES = [
    r"D:\data\DeChang\analysis_results_06_10\补充实验产出",
    r"D:\Documents\qwen-agent\02jflp9200\default\补充实验产出",
    os.path.join(os.getcwd(), "补充实验产出"),
]


def _resolve_out_dir():
    env_val = os.environ.get("DECHANG_OUT")
    if env_val:
        return env_val
    for c in _OUT_CANDIDATES:
        if os.path.isdir(c):
            return c
    return _OUT_CANDIDATES[0]


OUT_DIR = _resolve_out_dir()
print(f"[配置] 输出目录 = {OUT_DIR}")
PRED_PATH = os.path.join(OUT_DIR, "predictions_test.csv")
# 回退：基线阶段崩溃时 S1 已落盘核心结果 A–E
PRED_FALLBACK = os.path.join(OUT_DIR, "predictions_test_coreAE.csv")
if not os.path.exists(PRED_PATH) and os.path.exists(PRED_FALLBACK):
    print(f"⚠️ 未找到 predictions_test.csv，回退使用核心结果：{PRED_FALLBACK}")
    PRED_PATH = PRED_FALLBACK
ERR_PATH = os.path.join(OUT_DIR, "S2_error_series.csv")
PKL_PATH = os.path.join(OUT_DIR, "models_and_scaler.pkl")
SHAP_NPY = os.path.join(OUT_DIR, "S3_shap_values_sample.npy")
SHAP_X_NPY = os.path.join(OUT_DIR, "S3_X_val_sample.npy")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------- 统一绘图规范 ----------------------
DPI = 600
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})

# 四模型统一配色（全文一致）
COLORS = {
    "obs": "#000000",
    "A_raw_nwp": "#1f77b4",
    "B_lgbm": "#d62728",
    "D_lgbm_gqm": "#ff7f0e",
    "E_cascade": "#2ca02c",
    "F_randomforest": "#9467bd",
    "G_histgb": "#8c564b",
    "H_persistence": "#7f7f7f",
    "I_linear": "#17becf",
}
LABELS = {
    "obs": "Observed",
    "A_raw_nwp": "A: Raw NWP",
    "B_lgbm": "B: LightGBM (Stage 1)",
    "C_nwp_pcrqm": "C: NWP + PC-RQM",
    "D_lgbm_gqm": "D: LightGBM + Global QM",
    "E_cascade": "E: Cascade (Ours)",
    "F_randomforest": "F: Random Forest",
    "G_histgb": "G: HistGradientBoosting",
    "H_persistence": "H: Persistence",
    "I_linear": "I: Linear bias corr.",
}
LW = {"obs": 2.0, "A_raw_nwp": 1.5, "B_lgbm": 1.5,
      "D_lgbm_gqm": 1.5, "E_cascade": 2.0}


def save(fig, name):
    tif = os.path.join(FIG_DIR, f"{name}.tiff")
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(tif, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(png)
    plt.close(fig)
    print(f"  ✓ {name}  ->  {tif}")


# ==================== Fig 3：物理状态树 ====================
def fig_tree():
    if not os.path.exists(PKL_PATH):
        print("  ✗ Fig3_tree：缺少 models_and_scaler.pkl，跳过")
        return
    with open(PKL_PATH, "rb") as f:
        pack = pickle.load(f)
    tree = pack["regime_tree"]
    phys = pack["phys_cols"]

    t = tree.tree_
    n_nodes = t.node_count
    # 节点多时限制图幅，放大字号（原图标签过小的根因是 figsize 固定而节点多）
    depth = tree.get_depth()
    w = max(14, 2.2 * (2 ** min(depth, 3)))
    h = max(8, 2.2 * (depth + 1))

    fig, ax = plt.subplots(figsize=(w, h))
    from sklearn.tree import plot_tree
    plot_tree(tree, feature_names=phys, filled=True, rounded=True,
              fontsize=11, ax=ax, precision=3, impurity=False,
              proportion=False, max_depth=None)
    ax.set_title("Physically-Conditioned Regime Tree (CART, max_depth=3)",
                 fontsize=13, pad=16)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(9)
    save(fig, "Fig3_tree")

    # 叶节点统计表（比树图更易读，建议正文以表代图或图表并用）
    leaves = tree.apply  # noqa  (仅确认属性存在)
    rows = []
    for node_id in range(n_nodes):
        if t.children_left[node_id] == -1:      # 叶节点
            rows.append({
                "leaf_id": node_id,
                "n_samples": int(t.n_node_samples[node_id]),
                "mean_stage1_residual_m_s": round(float(t.value[node_id][0][0]), 4),
            })
    leaf_df = pd.DataFrame(rows).sort_values("n_samples", ascending=False)
    leaf_df.to_csv(os.path.join(FIG_DIR, "Fig3_leaf_table.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"  ✓ Fig3_leaf_table.csv（{len(leaf_df)} 个叶节点）")


# ==================== Fig 5：PDF 对比 + 差值子图 ====================
def fig_pdf(df):
    obs = df["obs"].values
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.12})

    keys = ["A_raw_nwp", "B_lgbm", "D_lgbm_gqm", "E_cascade"]
    keys = [k for k in keys if k in df.columns]
    styles = {"A_raw_nwp": "--", "B_lgbm": "-.", "D_lgbm_gqm": ":",
              "E_cascade": "-"}

    ax1.plot(np.sort(obs), stats.gaussian_kde(obs)(np.sort(obs)),
             color=COLORS["obs"], lw=LW["obs"], label=LABELS["obs"], zorder=5)
    xs = np.linspace(0, np.percentile(obs, 99.9), 600)
    for k in keys:
        v = df[k].values
        v = v[np.isfinite(v)]
        if v.size < 50:
            continue
        kde = stats.gaussian_kde(v)
        ax1.plot(xs, kde(xs), color=COLORS.get(k, "#333333"),
                 lw=LW.get(k, 1.5), ls=styles.get(k, "-"), label=LABELS[k])

    ax1.set_ylabel("Probability density")
    ax1.set_xlim(0, np.percentile(obs, 99.9) * 1.05)
    ax1.legend(loc="upper right", framealpha=0.92, ncol=1)
    ax1.set_title("(a) Probability density functions", loc="left", fontsize=12)

    # 差值子图：让定量差异可读（审稿人明确要求）
    kde_obs = stats.gaussian_kde(obs)
    base = kde_obs(xs)
    for k in ["E_cascade", "D_lgbm_gqm", "B_lgbm"]:
        if k not in df.columns:
            continue
        v = df[k].values
        v = v[np.isfinite(v)]
        if v.size < 50:
            continue
        ax2.plot(xs, stats.gaussian_kde(v)(xs) - base,
                 color=COLORS.get(k, "#333333"), lw=1.6,
                 ls=styles.get(k, "-"), label=f"{LABELS[k]} − Observed")
    ax2.axhline(0, color="k", lw=1.0, ls="-")
    ax2.set_xlabel("Wind speed (m/s)")
    ax2.set_ylabel("Density difference")
    ax2.legend(loc="upper left", framealpha=0.92, fontsize=8)
    ax2.set_title("(b) Residual density (model − observed)", loc="left", fontsize=11)
    ax2.yaxis.set_major_locator(MaxNLocator(5))

    save(fig, "Fig5_pdf")
    print("    ↳ 提示：若 (b) 中绿色曲线在 4–5 m/s 出现次峰、6.5–7.5 m/s 系统性偏正，"
          "须在 Limitations 说明为分区映射在边界处的副作用，"
          "并删除正文 “perfectly matches the ground truth”")


# ==================== Fig 6：大风过程时序 ====================
def fig_timeseries(df):
    if "obs_time" not in df.columns:
        print("  ✗ Fig6_timeseries：缺少 obs_time 列，跳过")
        return
    d = df.copy()
    d["obs_time"] = pd.to_datetime(d["obs_time"])
    d = d.sort_values("obs_time").drop_duplicates(subset=["obs_time"], keep="first")
    d = d.reset_index(drop=True)

    pos = int(np.argmax(d["obs"].values))
    half = 150                      # ±150 步 ≈ ±37.5 h（15 min 分辨率）
    lo, hi = max(0, pos - half), min(len(d), pos + half + 1)
    s = d.iloc[lo:hi]
    if len(s) < 10:
        print("  ✗ Fig6_timeseries：窗口样本不足，跳过")
        return

    fig, ax = plt.subplots(figsize=(13, 5.2))
    keys = ["A_raw_nwp", "B_lgbm", "E_cascade"]
    styles = {"A_raw_nwp": "--", "B_lgbm": "-.", "E_cascade": "-"}
    ax.plot(s["obs_time"], s["obs"], color=COLORS["obs"], lw=LW["obs"],
            label=LABELS["obs"], zorder=6)
    for k in keys:
        if k in s.columns:
            ax.plot(s["obs_time"], s[k], color=COLORS.get(k, "#333"),
                    lw=LW.get(k, 1.5), ls=styles.get(k, "-"),
                    label=LABELS[k], alpha=0.92)

    peak_t = s.iloc[pos - lo]["obs_time"] if pos - lo < len(s) else None
    if peak_t is not None:
        ax.axvline(pd.to_datetime(peak_t), color="gray", lw=1.0, ls=":")
        ax.annotate("peak observed wind", xy=(pd.to_datetime(peak_t), s["obs"].max()),
                    xytext=(12, -22), textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"))

    ax.set_ylabel("Wind speed (m/s)")
    ax.set_xlabel("Verification time (UTC)")
    ax.set_title("Time-series snapshot of a high-wind event", loc="left")
    ax.legend(loc="upper right", ncol=2, framealpha=0.92)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 标注 lead time 范围（审稿人要求交代时效）
    if "forecast_lead_time" in s.columns:
        lt_lo, lt_hi = s["forecast_lead_time"].min(), s["forecast_lead_time"].max()
        ax.text(0.01, 0.03, f"lead time in window: {lt_lo:.1f}–{lt_hi:.1f} h",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray",
                          alpha=0.85))
    save(fig, "Fig6_timeseries")
    


# ==================== Fig 7：SHAP ====================
def fig_shap():
    if not (os.path.exists(SHAP_NPY) and os.path.exists(SHAP_X_NPY)):
        print("  ✗ Fig7_shap：缺少 S3 保存的 SHAP 矩阵，跳过（请先运行 S3）")
        return
    try:
        import shap
    except ImportError:
        print("  ✗ Fig7_shap：未安装 shap，跳过")
        return
    sv = np.load(SHAP_NPY)
    Xv = np.load(SHAP_X_NPY)
    feats = ["forecast_wind_speed", "wind_shear", "richardson_number",
             "pressure_gradient", "temperature_gradient", "wind_dir_sin",
             "wind_dir_cos", "solar_radiation", "temp_wind_interaction",
             "pressure_wind_interaction", "season", "is_daytime", "month", "hour"]
    Xdf = pd.DataFrame(Xv, columns=feats[:Xv.shape[1]])
    if sv.ndim == 3:
        sv = sv[:, :, -1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, Xdf, max_display=14, show=False, plot_size=None)
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on predicted wind-speed residual)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("SHAP summary — Stage-1 LightGBM, computed on held-out validation set",
                 fontsize=12, loc="left", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Fig7_shap.tiff"), format="tiff",
                pil_kwargs={"compression": "tiff_lzw"}, dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "Fig7_shap.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ Fig7_shap")


# ==================== FigS：误差分布 + Q-Q ====================
def fig_error_dist(err_df):
    keys = [k for k in ["err_A_raw_nwp", "err_B_lgbm", "err_D_lgbm_gqm",
                        "err_E_cascade"] if k in err_df.columns]
    if not keys:
        print("  ✗ FigS_error_dist：缺少误差列，跳过")
        return
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(5.0 * ncol * 2, 3.4 * nrow),
                             squeeze=False)
    for i, k in enumerate(keys):
        r, c = divmod(i, ncol)
        ax_h = axes[r][c * 2]
        ax_q = axes[r][c * 2 + 1]
        e = err_df[k].values
        e = e[np.isfinite(e)]
        if e.size < 50:
            continue
        col = COLORS.get(k.replace("err_", ""), "#333333")
        name = LABELS.get(k.replace("err_", ""), k)

        ax_h.hist(e, bins=80, color=col, alpha=0.78, edgecolor="white", linewidth=0.3)
        ax_h.axvline(0, color="k", lw=1.0)
        ax_h.axvline(np.mean(e), color=col, lw=1.6, ls="--",
                     label=f"mean = {np.mean(e):+.2f}")
        ax_h.set_ylabel("Count")
        ax_h.set_xlabel("Error (m/s)")
        ax_h.set_title(f"{name}\nskew={stats.skew(e):.2f}, kurt={stats.kurtosis(e):.2f}",
                       fontsize=10, loc="left")
        ax_h.legend(fontsize=8, loc="upper right")

        stats.probplot(e, dist="norm", plot=ax_q)
        ax_q.get_lines()[0].set_markersize(2.0)
        ax_q.get_lines()[0].set_color(col)
        ax_q.get_lines()[1].set_color("k")
        ax_q.set_title(f"{name} · Normal Q–Q", fontsize=10, loc="left")
        ax_q.set_xlabel("Theoretical quantiles", fontsize=9)
        ax_q.set_ylabel("Sample quantiles", fontsize=9)

    for j in range(len(keys), nrow * ncol):
        r, c = divmod(j, ncol)
        axes[r][c * 2].axis("off")
        axes[r][c * 2 + 1].axis("off")
    fig.tight_layout()
    save(fig, "FigS_error_dist")


# ==================== FigS：性能随时效变化 ====================
def fig_leadtime(df):
    if "forecast_lead_time" not in df.columns:
        print("  ✗ FigS_leadtime：缺少 forecast_lead_time，跳过")
        return
    lt = df["forecast_lead_time"].values
    obs = df["obs"].values
    keys = [k for k in ["A_raw_nwp", "B_lgbm", "D_lgbm_gqm", "E_cascade"]
            if k in df.columns]
    if not keys:
        return
    # 按 6 h 一档分箱
    edges = np.arange(0, np.nanmax(lt) + 6, 6)
    centers, series = [], {k: [] for k in keys}
    vr_series = {k: [] for k in keys}
    for i in range(len(edges) - 1):
        m = (lt >= edges[i]) & (lt < edges[i + 1])
        if m.sum() < 30:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        o = obs[m]
        for k in keys:
            p = df[k].values[m]
            ok = np.isfinite(o) & np.isfinite(p)
            series[k].append(float(np.sqrt(np.mean((p[ok] - o[ok]) ** 2)))
                             if ok.sum() else np.nan)
            vo = np.var(o[ok])
            vr_series[k].append(float(np.var(p[ok]) / vo) if vo > 0 else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for k in keys:
        ax1.plot(centers, series[k], marker="o", ms=4, lw=1.8,
                 color=COLORS.get(k, "#333"), label=LABELS[k])
    ax1.set_xlabel("Forecast lead time (h)")
    ax1.set_ylabel("RMSE (m/s)")
    ax1.set_title("(a) RMSE vs lead time", loc="left")
    ax1.legend(fontsize=9)

    for k in keys:
        ax2.plot(centers, vr_series[k], marker="s", ms=4, lw=1.8,
                 color=COLORS.get(k, "#333"), label=LABELS[k])
    ax2.axhline(1.0, color="k", lw=1.0, ls="--")
    ax2.set_xlabel("Forecast lead time (h)")
    ax2.set_ylabel("Variance ratio  Var(pred)/Var(obs)")
    ax2.set_title("(b) Variance ratio vs lead time", loc="left")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "FigS_leadtime")


# ==================== FigS：Q-Q 曲线（分布还原证据） ====================
def fig_qq_curve(df):
    obs = np.sort(df["obs"].values)
    keys = [k for k in ["A_raw_nwp", "B_lgbm", "D_lgbm_gqm", "E_cascade"]
            if k in df.columns]
    if not keys:
        return
    qs = np.linspace(0.001, 0.999, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    lim = max(obs[-1], max(np.nanmax(df[k].values) for k in keys)) * 1.05
    ax1.plot([0, lim], [0, lim], "k--", lw=1.2, label="1:1")
    for k in keys:
        p = np.sort(df[k].values[np.isfinite(df[k].values)])
        ax1.plot(np.quantile(obs, qs), np.quantile(p, qs), lw=1.8,
                 color=COLORS.get(k, "#333"), label=LABELS[k])
    ax1.set_xlabel("Observed wind speed quantiles (m/s)")
    ax1.set_ylabel("Predicted wind speed quantiles (m/s)")
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
    ax1.set_title("(a) Quantile–quantile", loc="left")
    ax1.legend(fontsize=8)

    # 放大尾部：这是 R2-5/R2-8 关心的区间
    ax2.axhline(0, color="k", lw=1.0)
    for k in keys:
        p = df[k].values[np.isfinite(df[k].values)]
        oq, pq = np.quantile(obs, qs), np.quantile(np.sort(p), qs)
        ax2.plot(oq, pq - oq, lw=1.8, color=COLORS.get(k, "#333"), label=LABELS[k])
    ax2.set_xlabel("Observed wind speed quantile value (m/s)")
    ax2.set_ylabel("Predicted − observed quantile (m/s)")
    ax2.set_xlim(obs[int(0.85 * len(obs))], obs[-1])
    ax2.set_title("(b) Quantile error, upper 15% of the distribution", loc="left")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "FigS_qq_curve")


# ==================== FigS：分风速档 RMSE 带误差棒 ====================
def fig_bin_rmse(df):
    obs = df["obs"].values
    keys = [k for k in ["A_raw_nwp", "B_lgbm", "D_lgbm_gqm", "E_cascade"]
            if k in df.columns]
    edges = [0, 5, 10, 12, 15, np.inf]
    labels = ["[0,5)", "[5,10)", "[10,12)", "[12,15)", "[15,+∞)"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(keys), 1)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    for j, k in enumerate(keys):
        vals, lows, highs, ns = [], [], [], []
        for i in range(len(labels)):
            m = (obs >= edges[i]) & (obs < edges[i + 1])
            n = int(m.sum())
            ns.append(n)
            if n < 5:
                vals.append(np.nan); lows.append(0); highs.append(0)
                continue
            sub = df[m]
            r = float(np.sqrt(np.mean((sub[k].values - sub["obs"].values) ** 2)))
            # 按日分块 bootstrap 误差棒
            try:
                from S2_metrics_and_tests import block_bootstrap
                res = block_bootstrap(sub["date"].values, sub["obs"].values,
                                      sub[k].values, n_boot=300, stat="RMSE")
                lo, hi = res["A_CI"] if res else (np.nan, np.nan)
            except Exception:
                lo, hi = np.nan, np.nan
            vals.append(r)
            lows.append(r - lo if np.isfinite(lo) else 0)
            highs.append(hi - r if np.isfinite(hi) else 0)
        ax.bar(x + (j - len(keys) / 2 + 0.5) * width, vals, width,
               yerr=[lows, highs], capsize=3,
               color=COLORS.get(k, "#333"), label=LABELS[k],
               error_kw={"elinewidth": 0.9, "ecolor": "#444444"}, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\nN={n:,}" for lbl, n in zip(labels, ns)], fontsize=9)
    ax.set_xlabel("Observed wind speed bin (m/s)")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("RMSE by wind-speed bin with 95% block-bootstrap CI", loc="left")
    ax.legend(fontsize=9)
    save(fig, "FigS_bin_rmse")



# ==================== 主流程 ====================
def main():
    print("=" * 70)
    print("S4 · 出版级图件重绘（600 dpi TIFF + PNG）")
    print("=" * 70)

    df = None
    if os.path.exists(PRED_PATH):
        df = pd.read_csv(PRED_PATH)
        df["obs_time"] = pd.to_datetime(df["obs_time"])
        print(f"\n载入预测值：{len(df):,} 行")
    else:
        print(f"\n⚠️ 未找到 {PRED_PATH}，请先运行 S1")

    print("\n[1/8] Fig3 物理状态树")
    fig_tree()

    if df is not None:
        print("[2/8] Fig5 PDF 对比 + 差值子图")
        fig_pdf(df)
        print("[3/8] Fig6 大风过程时序")
        fig_timeseries(df)
    else:
        print("[2/8][3/8] 跳过（缺预测值）")

    print("[4/8] Fig7 SHAP")
    fig_shap()

    if os.path.exists(ERR_PATH):
        print("[5/8] FigS 误差分布 + Q-Q")
        fig_error_dist(pd.read_csv(ERR_PATH))
    else:
        print("[5/8] 跳过（缺 S2_error_series.csv，请先运行 S2）")

    if df is not None:
        print("[6/8] FigS 性能随时效变化")
        fig_leadtime(df)
        print("[7/8] FigS 分位数-分位数曲线")
        fig_qq_curve(df)
        print("[8/8] FigS 分风速档 RMSE 带误差棒")
        fig_bin_rmse(df)

    print("\n" + "=" * 70)
    print(f"全部图件输出目录：{FIG_DIR}")
    print("=" * 70)
    print("统一规范：600 dpi、坐标轴 11 pt、刻度 9 pt、图例 9 pt、"
          "四模型配色全文一致、分布图附差值子图。")
    print("投稿时 TIFF 用于正文，PNG 仅供自查预览。")


if __name__ == "__main__":
    main()
