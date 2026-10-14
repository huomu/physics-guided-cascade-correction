#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S2 · 补充指标与统计检验
=====================================================================
输入：S1 产出的 predictions_test.csv
输出：一组可直接誊入返修稿的 Markdown 表格 + CSV

覆盖的审稿意见
---------------------------------------------------------------------
R1-8 / R2-7  显著性检验、置信区间、误差分布分析、小样本不确定性
R2-5         Wasserstein / KS / 分位数误差 / 偏度峰度（方差比之外的分布证据）
R2-8         Model D(Global QM) vs Model E(PC-RQM) 的尾部与事件类指标对比
R2-6         点精度 ↔ 分布还原的权衡量化
R1-6         新增基线（RF / HistGB / Persistence / Linear）的完整对比
R2-12 / M11  分预报时效的性能表
R2-7         各风速档的「独立观测时刻数」与「独立事件天数」

依赖：pandas numpy scipy scikit-learn
运行：python S2_metrics_and_tests.py
=====================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp, skew, kurtosis
from scipy.interpolate import interp1d

# ---- 输出目录：与 S1 使用同一套自动探测逻辑，保证能读到 S1 的产出 ----
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
PRED_PATH = os.path.join(OUT_DIR, "predictions_test.csv")
# 回退：若 S1 在基线训练阶段崩溃，仍可用核心结果 A–E 的检查点文件
PRED_FALLBACK = os.path.join(OUT_DIR, "predictions_test_coreAE.csv")

N_BOOT = 1000            # block bootstrap 重复次数
RANDOM_STATE = 42
EVENT_THRESHOLDS = [10.0, 12.0, 15.0]     # POD/FAR/TS 阈值

# 模型列 -> 显示名（与论文表4的 A–E 命名保持一致，F–I 为新增基线）
MODELS = [
    ("A_raw_nwp",      "A: Raw NWP"),
    ("B_lgbm",         "B: LightGBM (Stage 1)"),
    ("C_nwp_pcrqm",    "C: NWP + PC-RQM (Stage 2 only)"),
    ("D_lgbm_gqm",     "D: LightGBM + Global QM"),
    ("E_cascade",      "E: Cascade (Ours)"),
    ("F_randomforest", "F: Random Forest"),
    ("G_histgb",       "G: HistGradientBoosting"),
    ("H_persistence",  "H: Persistence (t−15min)"),
    ("I_linear",       "I: Linear bias correction"),
]

# 原论文表3分档（用于复现）与扩展分档（含 12 m/s，作为补充）
BINS_ORIG = ([0, 5, 10, 15, np.inf], ["[0,5)", "[5,10)", "[10,15)", "[15,+∞)"])
BINS_EXT = ([0, 5, 10, 12, 15, np.inf],
            ["[0,5)", "[5,10)", "[10,12)", "[12,15)", "[15,+∞)"])
LT_BANDS = {"0–24 h": (0, 24), "24–48 h": (24, 48), "48–72 h": (48, 72),
            ">72 h": (72, np.inf)}


# ============================ 基础指标 ============================
def det_metrics(obs, pred):
    """确定性指标 + 方差比 + 形状量。"""
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]
    n = obs.size
    if n == 0:
        return None
    err = pred - obs
    var_o, var_p = float(np.var(obs)), float(np.var(pred))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    q_o = np.percentile(obs, [50, 90, 95, 99])
    q_p = np.percentile(pred, [50, 90, 95, 99])
    return {
        "N": n,
        "MBE": float(err.mean()),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "VarRatio": float(var_p / var_o) if var_o > 0 else np.nan,
        "Skew_obs": float(skew(obs)), "Skew_pred": float(skew(pred)),
        "Kurt_obs": float(kurtosis(obs)), "Kurt_pred": float(kurtosis(pred)),
        "W1": float(wasserstein_distance(obs, pred)),
        "KS_stat": float(ks_2samp(obs, pred).statistic),
        "KS_p": float(ks_2samp(obs, pred).pvalue),
        "Q50_err": float(q_p[0] - q_o[0]),
        "Q90_err": float(q_p[1] - q_o[1]),
        "Q95_err": float(q_p[2] - q_o[2]),
        "Q99_err": float(q_p[3] - q_o[3]),
    }


def event_scores(obs, pred, thr):
    """事件类指标 POD / FAR / TS / CSI（回应 R2-8：尾部与事件视角）。"""
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]
    hit = float(np.sum((obs >= thr) & (pred >= thr)))
    miss = float(np.sum((obs >= thr) & (pred < thr)))
    fa = float(np.sum((obs < thr) & (pred >= thr)))
    tn = float(np.sum((obs < thr) & (pred < thr)))
    return {
        "Threshold": thr, "N_events_obs": int(hit + miss),
        "Hit": int(hit), "Miss": int(miss), "FalseAlarm": int(fa),
        "POD": hit / (hit + miss) if (hit + miss) > 0 else np.nan,
        "FAR": fa / (hit + fa) if (hit + fa) > 0 else np.nan,
        "TS": hit / (hit + miss + fa) if (hit + miss + fa) > 0 else np.nan,
        "HSS": (2 * (hit * tn - fa * miss) /
                ((hit + miss) * (miss + tn) + (hit + fa) * (fa + tn)))
        if ((hit + miss) * (miss + tn) + (hit + fa) * (fa + tn)) > 0 else np.nan,
    }


# ==================== 按日分块的 block bootstrap ====================
def make_day_blocks(dates):
    """按日历日分块。返回 (block索引数组列表, 块数)。"""
    dates = pd.to_datetime(pd.Series(np.asarray(dates))).dt.date.values
    uniq = sorted(set(dates))
    blocks = [np.where(dates == d)[0] for d in uniq]
    return blocks, len(blocks)


def block_bootstrap(dates, obs, pred_a, pred_b=None, n_boot=N_BOOT,
                    seed=RANDOM_STATE, stat="RMSE"):
    """按日分块的（配对）bootstrap。
    pred_b=None 时返回 pred_a 的统计量 CI；否则同时返回配对差值 (B−A) 的 CI 与
    单侧 p 值。分块保留时序自相关，避免用行级 iid bootstrap 虚高显著性。"""
    blocks, nb = make_day_blocks(dates)
    if nb < 4:
        return None
    obs = np.asarray(obs, float)
    pred_a = np.asarray(pred_a, float)
    pred_b = None if pred_b is None else np.asarray(pred_b, float)
    rng = np.random.default_rng(seed)

    def stat_of(o, p):
        if stat == "RMSE":
            return float(np.sqrt(np.mean((p - o) ** 2)))
        if stat == "MAE":
            return float(np.mean(np.abs(p - o)))
        if stat == "VarRatio":
            vo = np.var(o)
            return float(np.var(p) / vo) if vo > 0 else np.nan
        if stat == "W1":
            return float(wasserstein_distance(o, p))
        if stat == "R2":
            ss = np.sum((o - o.mean()) ** 2)
            return float(1 - np.sum((p - o) ** 2) / ss) if ss > 0 else np.nan
        raise ValueError(stat)

    sa, sb, diff = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, nb, size=nb)
        idx = np.concatenate([blocks[i] for i in pick])
        o = obs[idx]
        va = stat_of(o, pred_a[idx])
        sa.append(va)
        if pred_b is not None:
            vb = stat_of(o, pred_b[idx])
            sb.append(vb)
            diff.append(vb - va)

    def ci(arr):
        a = np.asarray(arr, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan)
        a = np.sort(a)
        return (float(a[int(0.025 * a.size)]), float(a[int(0.975 * a.size)]))

    res = {"stat": stat, "n_blocks": nb,
           "A_CI": ci(sa), "A_mean": float(np.nanmean(sa))}
    if pred_b is not None:
        d = np.asarray(diff, float)
        d = d[np.isfinite(d)]
        res["B_CI"] = ci(sb)
        res["B_mean"] = float(np.nanmean(sb))
        res["Diff_CI"] = ci(diff)
        res["Diff_mean"] = float(np.nanmean(d))
        # 单侧：差值(B−A) ≥ 0 的比例，即"B 不比 A 好"的概率
        res["p_B_not_better"] = float(np.mean(d >= 0)) if d.size else np.nan
        res["significant"] = bool(res["Diff_CI"][0] > 0 or res["Diff_CI"][1] < 0)
    return res


def fmt(v, nd=3):
    try:
        return "N/A" if v is None or not np.isfinite(float(v)) else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "N/A"


# ============================ 主流程 ============================
def main():
    global PRED_PATH
    if not os.path.exists(PRED_PATH):
        if os.path.exists(PRED_FALLBACK):
            print("⚠️ 未找到 predictions_test.csv，回退使用核心结果检查点文件：")
            print(f"   {PRED_FALLBACK}")
            print("   （缺少 F/G/H/I 四个基线列，相关行将自动跳过）")
            PRED_PATH = PRED_FALLBACK
        else:
            raise FileNotFoundError(
                f"找不到 {PRED_PATH}\n请先运行 S1_run_and_export.py")

    df = pd.read_csv(PRED_PATH)
    df["obs_time"] = pd.to_datetime(df["obs_time"])
    print(f"载入预测值：{len(df):,} 行")

    # 只保留实际存在的模型列，避免后续按名称索引时报 KeyError
    present = [(c, n) for c, n in MODELS if c in df.columns]
    missing = [n for c, n in MODELS if c not in df.columns]
    if missing:
        print(f"⚠️ 以下模型列缺失，将被跳过：{missing}")
    MODELS_ACTIVE = present

    obs = df["obs"].values
    dates = df["date"].values
    lt = df["forecast_lead_time"].values

    report = []
    A = report.append
    A("# 补充指标与统计检验结果（S2）")
    A("")
    A("> 本文件由 `S2_metrics_and_tests.py` 自动生成。所有数字基于 **S1 复现的原实验"
      "预测值**，未重训模型、未改动验证协议，论文表 2/3/4 的原有数字不变。")
    A("")
    A(f"- 测试集样本数（配对行）：**{len(df):,}**")
    A(f"- 其中独立观测时刻数：**{df['obs_time'].nunique():,}**"
      f"（重复倍数 {len(df) / max(df['obs_time'].nunique(), 1):.2f}）")
    A(f"- 覆盖日历日数：**{df['date'].nunique()}**")
    A(f"- lead time 范围：**{np.nanmin(lt):.2f} – {np.nanmax(lt):.2f} h**")
    A(f"- block bootstrap：按日历日分块，{N_BOOT} 次重采样")
    A("")

    # ---------- 表 S1：全套确定性 + 分布指标 ----------
    A("## 表 S1 · 全套指标：确定性 + 方差比 + 分布形状 + 分位数误差")
    A("")
    A("（回应 R1-6 基线扩展、R2-5 分布类指标、R2-6 权衡量化、R2-8 分区增量价值）")
    A("")
    A("| 模型 | N | MBE | RMSE | MAE | R² | 方差比 | Skew | Kurt | "
      "Wasserstein | KS | KS p | ΔQ90 | ΔQ95 | ΔQ99 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    all_m = {}
    for col, name in MODELS:
        if col not in df.columns:
            continue
        m = det_metrics(obs, df[col].values)
        if m is None:
            continue
        all_m[name] = m
        A(f"| {name} | {m['N']:,} | {fmt(m['MBE'])} | {fmt(m['RMSE'])} | "
          f"{fmt(m['MAE'])} | {fmt(m['R2'])} | {fmt(m['VarRatio'])} | "
          f"{fmt(m['Skew_pred'], 2)} | {fmt(m['Kurt_pred'], 2)} | "
          f"{fmt(m['W1'], 4)} | {fmt(m['KS_stat'], 4)} | "
          f"{m['KS_p']:.2e} | {fmt(m['Q90_err'])} | {fmt(m['Q95_err'])} | "
          f"{fmt(m['Q99_err'])} |")
    A("")
    ref = det_metrics(obs, obs)
    A(f"观测分布参考量：Skew(obs) = {fmt(ref['Skew_obs'], 3)}，"
      f"Kurt(obs) = {fmt(ref['Kurt_obs'], 3)}")
    A("")
    A("> 判读要点：方差比只约束二阶矩。要主张「还原了观测分布」，需同时看 "
      "Wasserstein（整体分布距离）、KS（最大累积概率差）、ΔQ90/ΔQ95/ΔQ99"
      "（尾部分位数误差）与 Skew/Kurt 是否接近观测量。")
    A("")

    # ---------- 表 S2：显著性检验 ----------
    A("## 表 S2 · 配对显著性检验（按日分块 block bootstrap）")
    A("")
    A("（回应 R1-8 统计检验、R2-6/R2-8 权衡的显著性）")
    A("")
    A("差值定义 = 统计量(后者) − 统计量(前者)。对 RMSE，差值 < 0 表示后者更优；"
      "95% CI 不含 0 即差异显著。")
    A("")
    pairs = [
        ("B: LightGBM (Stage 1)", "A: Raw NWP", "B vs A"),
        ("E: Cascade (Ours)", "B: LightGBM (Stage 1)", "E vs B（核心权衡）"),
        ("E: Cascade (Ours)", "D: LightGBM + Global QM", "E vs D（分区增量）"),
        ("D: LightGBM + Global QM", "B: LightGBM (Stage 1)", "D vs B（QM 的代价）"),
        ("F: Random Forest", "B: LightGBM (Stage 1)", "F vs B"),
        ("G: HistGradientBoosting", "B: LightGBM (Stage 1)", "G vs B"),
        ("H: Persistence (t−15min)", "A: Raw NWP", "H vs A"),
        ("I: Linear bias correction", "A: Raw NWP", "I vs A"),
    ]
    # 只映射实际存在的列：若基线缺失，对应配对会自动跳过而不是抛 KeyError
    col_of = {name: col for col, name in MODELS if col in df.columns}

    for stat in ["RMSE", "MAE", "VarRatio", "W1"]:
        A(f"### 统计量 = {stat}")
        A("")
        A("| 对比 | 前者 | 后者 | 差值均值 | 差值 95% CI | 显著? | "
          "单侧 p（后者不更优） | 块数 |")
        A("|---|---|---|---|---|---|---|---|")
        for a_name, b_name, label in pairs:
            if a_name not in col_of or b_name not in col_of:
                continue
            pa = df[col_of[a_name]].values
            pb = df[col_of[b_name]].values
            r = block_bootstrap(dates, obs, pa, pb, stat=stat)
            if r is None:
                A(f"| {label} | — | — | — | — | — | — | — |")
                continue
            d_lo, d_hi = r["Diff_CI"]
            A(f"| {label} | {fmt(r['A_mean'], 4)} | {fmt(r['B_mean'], 4)} | "
              f"{fmt(r['Diff_mean'], 4)} | [{fmt(d_lo, 4)}, {fmt(d_hi, 4)}] | "
              f"{'✅ 是' if r['significant'] else '❌ 否'} | "
              f"{fmt(r['p_B_not_better'], 3)} | {r['n_blocks']} |")
        A("")

    # 每个模型的 RMSE 自身 CI
    A("### 各模型 RMSE 的 95% CI")
    A("")
    A("| 模型 | RMSE 点估计 | RMSE 95% CI | 块数 |")
    A("|---|---|---|---|")
    for col, name in MODELS:
        if col not in df.columns:
            continue
        r = block_bootstrap(dates, obs, df[col].values, stat="RMSE")
        if r is None:
            continue
        A(f"| {name} | {fmt(r['A_mean'])} | "
          f"[{fmt(r['A_CI'][0])}, {fmt(r['A_CI'][1])}] | {r['n_blocks']} |")
    A("")

    # ---------- 表 S3：事件类指标 ----------
    A("## 表 S3 · 事件类指标 POD / FAR / TS / HSS")
    A("")
    A("（回应 R2-8：用事件与尾部视角证明物理分区条件化的附加价值）")
    A("")
    for thr in EVENT_THRESHOLDS:
        A(f"### 阈值 ≥ {thr:.0f} m/s")
        A("")
        A("| 模型 | 观测事件数 | Hit | Miss | FalseAlarm | POD | FAR | TS | HSS |")
        A("|---|---|---|---|---|---|---|---|---|")
        for col, name in MODELS:
            if col not in df.columns:
                continue
            e = event_scores(obs, df[col].values, thr)
            A(f"| {name} | {e['N_events_obs']:,} | {e['Hit']:,} | {e['Miss']:,} | "
              f"{e['FalseAlarm']:,} | {fmt(e['POD'])} | {fmt(e['FAR'])} | "
              f"{fmt(e['TS'])} | {fmt(e['HSS'])} |")
        A("")

    # ---------- 表 S4：分风速档（含独立事件数） ----------
    for tag, (edges, labels) in [("原分档（复现论文表 3）", BINS_ORIG),
                                 ("扩展分档（含 12 m/s）", BINS_EXT)]:
        A(f"## 表 S4 · 分风速段指标 — {tag}")
        A("")
        A("「独立观测时刻」= 该档内 obs_time 去重后的个数；「独立事件天数」= 涉及的不同"
          "日历日数。二者用于揭示配对行重复计数带来的有效样本量差异（回应 R2-7）。")
        A("")
        A("| 风速档 | 配对行数 | 独立观测时刻 | 独立事件天数 | 模型 | RMSE | RMSE 95% CI |")
        A("|---|---|---|---|---|---|---|")
        for i, lbl in enumerate(labels):
            lo, hi = edges[i], edges[i + 1]
            mask = (obs >= lo) & (obs < hi)
            n_rows = int(mask.sum())
            n_uniq = int(df.loc[mask, "obs_time"].nunique())
            n_days = int(df.loc[mask, "date"].nunique())
            if n_rows == 0:
                A(f"| {lbl} | 0 | 0 | 0 | — | N/A | N/A |")
                continue
            for col, name in MODELS:
                if col not in df.columns:
                    continue
                sub = df[mask]
                r = block_bootstrap(sub["date"].values, sub["obs"].values,
                                    sub[col].values, stat="RMSE")
                rmse_v = fmt(np.sqrt(np.mean((sub[col].values - sub["obs"].values) ** 2)))
                ci = f"[{fmt(r['A_CI'][0])}, {fmt(r['A_CI'][1])}]" if r else "N/A"
                head = (f"| {lbl} | {n_rows:,} | {n_uniq:,} | {n_days} "
                        if col == MODELS[0][0] else "| | | | ")
                A(f"{head}| {name} | {rmse_v} | {ci} |")
        A("")
        A("> ⚠️ 若某档「独立事件天数」为 1，则该档不构成统计证据，正文须标注为单例个案"
          "并撤下改善性表述。")
        A("")

    # ---------- 表 S5：分预报时效 ----------
    A("## 表 S5 · 分预报时效性能（回应 R2-12 / M11）")
    A("")
    A("> 原实验未做时效过滤，混合了 0.25 h 至最长时效的全部样本。下表按时效分组，"
      "使性能随时效的衰减可见；注意各组样本为同一批观测时刻在不同时效的预报，"
      "**不可跨组相加**。")
    A("")
    for band, (lo, hi) in LT_BANDS.items():
        mask = (lt >= lo) & (lt < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        sub = df[mask]
        A(f"### {band}（N = {n:,}，独立观测时刻 {sub['obs_time'].nunique():,}）")
        A("")
        A("| 模型 | N | MBE | RMSE | MAE | R² | 方差比 | Wasserstein | ΔQ95 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for col, name in MODELS:
            if col not in sub.columns:
                continue
            m = det_metrics(sub["obs"].values, sub[col].values)
            if m is None:
                continue
            A(f"| {name} | {m['N']:,} | {fmt(m['MBE'])} | {fmt(m['RMSE'])} | "
              f"{fmt(m['MAE'])} | {fmt(m['R2'])} | {fmt(m['VarRatio'])} | "
              f"{fmt(m['W1'], 4)} | {fmt(m['Q95_err'])} |")
        A("")

    # ---------- 表 S6：误差分布诊断 ----------
    A("## 表 S6 · 误差分布诊断（回应 R1-8 误差分布分析）")
    A("")
    A("| 模型 | 误差均值 | 误差标准差 | 误差偏度 | 误差峰度 | 误差 P5 | 误差 P95 | "
      "\|误差\|>2 m/s 占比 | \|误差\|>4 m/s 占比 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for col, name in MODELS:
        if col not in df.columns:
            continue
        e = df[col].values - obs
        e = e[np.isfinite(e)]
        if e.size == 0:
            continue
        A(f"| {name} | {fmt(e.mean())} | {fmt(e.std())} | {fmt(skew(e), 3)} | "
          f"{fmt(kurtosis(e), 3)} | {fmt(np.percentile(e, 5))} | "
          f"{fmt(np.percentile(e, 95))} | "
          f"{100 * np.mean(np.abs(e) > 2):.1f}% | "
          f"{100 * np.mean(np.abs(e) > 4):.1f}% |")
    A("")

    # ---------- 权衡汇总 ----------
    A("## 表 S7 · 点精度 ↔ 分布还原权衡汇总（回应 R2-6）")
    A("")
    A("以 B（LightGBM）为参照，正号表示相对 B 的恶化/增大。")
    A("")
    A("| 模型 | ΔRMSE vs B | ΔMAE vs B | ΔR² vs B | Δ方差比 vs B（越接近0越好） | "
      "ΔW1 vs B（负=更优） | ΔQ99 vs B |")
    A("|---|---|---|---|---|---|---|")
    if "B: LightGBM (Stage 1)" in all_m:
        b = all_m["B: LightGBM (Stage 1)"]
        for name, m in all_m.items():
            if name == "B: LightGBM (Stage 1)":
                continue
            d_vr = m["VarRatio"] - b["VarRatio"]
            A(f"| {name} | {m['RMSE'] - b['RMSE']:+.3f} | "
              f"{m['MAE'] - b['MAE']:+.3f} | {m['R2'] - b['R2']:+.3f} | "
              f"{d_vr:+.3f}（\|·\|={abs(d_vr):.3f}） | "
              f"{m['W1'] - b['W1']:+.4f} | {m['Q99_err'] - b['Q99_err']:+.3f} |")
    A("")

    # ---------- 导出 ----------
    md_path = os.path.join(OUT_DIR, "S2_补充指标与统计检验.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n报告已写入：{md_path}")

    # 同步导出机器可读版，便于自行作图
    rows = []
    for col, name in MODELS:
        if col in df.columns:
            m = det_metrics(obs, df[col].values)
            if m:
                rows.append({"Model": name, **m})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "S2_overall_metrics.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"指标表已导出：{os.path.join(OUT_DIR, 'S2_overall_metrics.csv')}")

    # 误差序列导出（供 S4 绘制误差分布图 / Q-Q 图）
    err_df = df[["obs_time", "date", "forecast_lead_time"]].copy()
    for col, name in MODELS:
        if col in df.columns:
            err_df[f"err_{col}"] = df[col].values - obs
            err_df[f"pred_{col}"] = df[col].values
    err_df["obs"] = obs
    p = os.path.join(OUT_DIR, "S2_error_series.csv")
    err_df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"误差序列已导出：{p}")
    print("\n下一步：python S3_shap_stability_latency.py")


if __name__ == "__main__":
    main()
