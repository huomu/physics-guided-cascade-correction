#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S1 · 复现原实验 + 补充基线模型 + 导出预测值
=====================================================================
【设计原则】本脚本**不改动**原实验的任何设定：
  · 数据切分：完全复现原代码的「每月内部按时间 70/15/15」交错切分
  · Stage-1 ：LightGBM 原参数（800棵/lr0.03/depth7/leaves31/subsample0.8）
  · Stage-2 ：CART(max_depth=3, min_samples_leaf=500) + 1000分位点
              interp1d(fill_value="extrapolate")，与原实现逐行一致
  · 数据口径：全量 lead time（不做 ≤72h 过滤），与论文表2/表3一致
因此表2、表3、表4 的数字应当**逐格复现**，脚本内置自检会打印比对结果。

【新增内容】仅在此之上增加 4 个基线模型（回应 R1-6）：
  F: Random Forest            G: HistGradientBoosting
  H: Persistence (t-15min)    I: Linear bias correction (回归式)
以及 Model C / Model D（复现表4消融）。

【产出】predictions_full.csv —— 后续 S2/S3 全部指标计算的唯一输入。

依赖：pandas numpy scikit-learn lightgbm scipy
运行：python S1_run_and_export.py
=====================================================================
"""

import os

# ============================================================================
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 允许重复加载 OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "4")            # 限制线程数，减少争用
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")       # 避免 fork 时重复初始化

import gc
import sys
import time
import warnings
import subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import interp1d

# 注意：HistGradientBoostingRegressor **不在这里导入**。
# 它是触发 OpenMP 冲突的元凶，改为在独立子进程中训练（见 _train_hgb_in_subprocess）。

warnings.filterwarnings("ignore")

# ============================ 配置区 ============================

_CSV_CANDIDATES = [
    r"D:\data\analysis_results_06_10\paired_data.csv",   # 本机实验目录
    r"C:\Users\admin\Desktop\analysis_results_06_10\paired_data.csv",  # 桌面副本
    r"C:\Users\admin\Desktop\paired_data.csv",                   # 桌面根目录
    os.path.join(os.getcwd(), "paired_data.csv"),                 # 当前工作目录
]
_OUT_CANDIDATES = [
    r"D:\data\DeChang\analysis_results_06_10\补充实验产出",
    r"D:\Documents\qwen-agent\02jflp9200\default\补充实验产出",
    os.path.join(os.getcwd(), "补充实验产出"),
]


def _first_existing(candidates, env_key=None, must_exist=True):
    """返回候选列表中第一个存在（或可用于创建）的路径。"""
    if env_key:
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


CSV_PATH = _first_existing(_CSV_CANDIDATES, env_key="DECHANG_CSV", must_exist=True)
OUT_DIR = _first_existing(_OUT_CANDIDATES, env_key="DECHANG_OUT", must_exist=False)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[配置] 数据文件 = {CSV_PATH}")
print(f"[配置] 输出目录 = {OUT_DIR}")

RANDOM_STATE = 42
N_QM_POINTS = 1000          
CART_MAX_DEPTH = 3
CART_MIN_SAMPLES_LEAF = 500  

# ---- 基线模型的资源配置 ----
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None          # None = 标准 RF，不限深度
RF_MIN_SAMPLES_LEAF = 1      # 标准 RF 默认值
RF_MAX_SAMPLES = None        # None = 每棵树使用完整 bootstrap 抽样（标准行为）
RF_MAX_FEATURES = 1.0        # 回归任务标准默认：每次分裂考虑全部特征
RF_N_JOBS = 4                # 并行树数，按本机 CPU 核数调整即可（与内存无关）
HGB_MAX_ITER = 300
HGB_MAX_LEAF_NODES = 31      # 与 Stage-1 LightGBM 容量对齐，保证基线间公平比较

# 与原实现完全一致的特征列顺序
FEATURE_COLS = [
    "forecast_wind_speed", "wind_shear", "richardson_number",
    "pressure_gradient", "temperature_gradient", "wind_dir_sin",
    "wind_dir_cos", "solar_radiation", "temp_wind_interaction",
    "pressure_wind_interaction", "season", "is_daytime", "month", "hour",
]
# 与原实现完全一致的纯物理特征（仅用于 CART 分区）
PHYS_COLS = [
    "richardson_number", "temperature_gradient", "wind_shear",
    "pressure_gradient", "temp_wind_interaction",
]

# 论文原表数字，用于自检比对
PAPER_TABLE2 = {
    "Raw NWP":  {"MBE": 1.803, "RMSE": 4.165, "MAE": 3.207, "R2": -1.843},
    "LightGBM": {"MBE": 0.042, "RMSE": 2.076, "MAE": 1.618, "R2": 0.294},
    "Cascade":  {"MBE": 0.045, "RMSE": 2.432, "MAE": 1.898, "R2": 0.030},
}
PAPER_TABLE3_N = [37055, 27661, 2191, 13]   # [0,5) [5,10) [10,15) [15,+inf)


# ============================ 工具函数 ============================
def metrics(obs, pred):
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]
    if obs.size == 0:
        return {"N": 0}
    err = pred - obs
    return {
        "N": int(obs.size),
        "MBE": float(err.mean()),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(r2_score(obs, pred)),
    }


def empirical_qm_extrapolate(src_train, tgt_train, src_test, n_points=N_QM_POINTS):
    """与原实现逐行一致的经验分位数映射（线性外推版）。"""
    src_train = np.asarray(src_train, float)
    tgt_train = np.asarray(tgt_train, float)
    src_test = np.asarray(src_test, float)
    if src_test.size == 0 or src_train.size < 10:
        return src_test.copy()
    qs = np.linspace(0, 100, n_points)
    src_q = np.percentile(src_train, qs)
    tgt_q = np.percentile(tgt_train, qs)
    # np.unique 保证 interp1d 的 x 严格递增（原实现在分位点重复时会抛错或产生非单调插值）
    src_q, uidx = np.unique(src_q, return_index=True)
    tgt_q = tgt_q[uidx]
    if src_q.size < 2:
        return src_test.copy()
    mapping = interp1d(src_q, tgt_q, kind="linear", bounds_error=False,
                       fill_value="extrapolate")
    return np.clip(mapping(src_test), 0, None)


def global_qm_extrapolate(src_train, tgt_train, src_test):
    """Model D：不分流态的全局分位数映射。"""
    return empirical_qm_extrapolate(src_train, tgt_train, src_test)


# ==================== G 基线的子进程隔离训练 ====================
# 隔离方案：
#   主进程把训练/测试矩阵写入临时 npz，启动一个**只导入 sklearn**的干净子进程
#   完成 HGB 训练与预测，再把预测结果写回 npz。子进程崩溃只影响 G 这一列。

_HGB_CHILD_SOURCE = r'''
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import sys
import numpy as np

work = sys.argv[1]
max_iter = int(sys.argv[2])
max_leaf = int(sys.argv[3])
seed = int(sys.argv[4])

d = np.load(os.path.join(work, "hgb_input.npz"))
X_tr, y_tr = d["X_train"], d["y_train"]
X_te, fcst_te = d["X_test"], d["fcst_test"]

from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    max_iter=max_iter, learning_rate=0.05,
    max_depth=7, max_leaf_nodes=max_leaf,
    min_samples_leaf=20, l2_regularization=0.1,
    early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=30, random_state=seed)
model.fit(X_tr, y_tr)

pred = np.clip(fcst_te + model.predict(X_te), 0, None)
np.savez(os.path.join(work, "hgb_output.npz"),
         pred=pred, n_iter=int(model.n_iter_))
print("HGB_CHILD_OK n_iter=%d" % int(model.n_iter_))
'''


def _train_hgb_in_subprocess(X_train_s, y_train, X_test_s, fcst_test, out_dir,
                             max_iter=300, max_leaf_nodes=31,
                             random_state=42, timeout=3600):
    """在独立子进程中训练 HistGradientBoosting 并取回预测值。

    返回 (pred 或 None, n_iter, 错误信息)。任何失败都不抛异常，
    以保证 A–F、H、I 的结果不被连累。
    """
    import tempfile
    import shutil

    work = tempfile.mkdtemp(prefix="hgb_child_")
    try:
        np.savez(os.path.join(work, "hgb_input.npz"),
                 X_train=np.asarray(X_train_s, dtype=np.float64),
                 y_train=np.asarray(y_train, dtype=np.float64),
                 X_test=np.asarray(X_test_s, dtype=np.float64),
                 fcst_test=np.asarray(fcst_test, dtype=np.float64))

        child = os.path.join(work, "hgb_child.py")
        with open(child, "w", encoding="utf-8") as f:
            f.write(_HGB_CHILD_SOURCE)

        proc = subprocess.run(
            [sys.executable, child, work, str(max_iter),
             str(max_leaf_nodes), str(random_state)],
            capture_output=True, text=True, timeout=timeout)

        out_path = os.path.join(work, "hgb_output.npz")
        if proc.returncode != 0 or not os.path.exists(out_path):
            err = (proc.stderr or proc.stdout or "").strip()
            code = proc.returncode
            hint = ""
            if code == -1073740793 or "C06D007F" in err.upper() or code == 3228369023:
                hint = "（0xC06D007F：OpenMP 运行时冲突，建议 conda install -c conda-forge nomkl 或改用 intel-openmp 单一版本）"
            return None, None, f"子进程退出码 {code} {hint}\n{err[-1500:]}"

        d = np.load(out_path)
        return d["pred"], int(d["n_iter"]), None
    except subprocess.TimeoutExpired:
        return None, None, f"子进程超时（>{timeout}s）"
    except Exception as exc:                      # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ============================ 主流程 ============================
def main():
    t_start = time.time()

    # ---------- [1/8] 加载数据 ----------
    print("\n" + "=" * 70)
    print("[1/8] 加载数据")
    print("=" * 70)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到数据文件：{CSV_PATH}\n请修改脚本顶部 CSV_PATH。")
    df = pd.read_csv(CSV_PATH)
    df["obs_time"] = pd.to_datetime(df["obs_time"])
    df = df.sort_values("obs_time").reset_index(drop=True)
    print(f"  总行数 = {len(df):,}")

    missing = [c for c in FEATURE_COLS + PHYS_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少特征列：{missing}")

    # 目标变量：加法残差（与原实现一致）
    df["wind_speed_error"] = df["obs_wind_speed"] - df["forecast_wind_speed"]
    df["year_month"] = df["obs_time"].dt.to_period("M")
    df["date"] = df["obs_time"].dt.date

    # ---------- [2/8] 复现原切分：每月内部 70/15/15 ----------
    print("\n" + "=" * 70)
    print("[2/8] 数据切分（复现原实现的每月内部 70/15/15）")
    print("=" * 70)
    train_idx, val_idx, test_idx = [], [], []
    for ym, group in df.groupby("year_month"):
        n = len(group)
        tr_end = int(n * 0.70)
        va_end = int(n * 0.85)
        idx = group.index.values          # 已按 obs_time 排序
        train_idx.extend(idx[:tr_end])
        val_idx.extend(idx[tr_end:va_end])
        test_idx.extend(idx[va_end:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    print(f"  训练集 = {len(train_idx):,} | 验证集 = {len(val_idx):,} | "
          f"测试集 = {len(test_idx):,}")
    print(f"  测试集占比 = {len(test_idx) / len(df):.4f}")

    X_train = df.loc[train_idx, FEATURE_COLS]
    X_val = df.loc[val_idx, FEATURE_COLS]
    X_test = df.loc[test_idx, FEATURE_COLS]
    y_train = df.loc[train_idx, "wind_speed_error"].values
    y_val = df.loc[val_idx, "wind_speed_error"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    obs_train = df.loc[train_idx, "obs_wind_speed"].values
    fcst_train = df.loc[train_idx, "forecast_wind_speed"].values
    obs_test = df.loc[test_idx, "obs_wind_speed"].values
    fcst_test = df.loc[test_idx, "forecast_wind_speed"].values

    # ---------- [3/8] Stage 1: LightGBM（原参数） ----------
    print("\n" + "=" * 70)
    print("[3/8] Stage 1 · LightGBM 残差订正（原参数）")
    print("=" * 70)
    t0 = time.time()
    model_lgb = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=7, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        objective="regression",
    )
    model_lgb.fit(X_train_s, y_train,
                  eval_set=[(X_val_s, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
    lat_lgb_train = (time.time() - t0)
    print(f"  训练耗时 {lat_lgb_train:.1f}s，best_iteration = "
          f"{getattr(model_lgb, 'best_iteration_', 'n/a')}")

    lgbm_corrected_train = np.clip(fcst_train + model_lgb.predict(X_train_s), 0, None)
    lgbm_corrected_test = np.clip(fcst_test + model_lgb.predict(X_test_s), 0, None)

    # ---------- [4/8] Stage 2: PC-RQM（原实现） ----------
    print("\n" + "=" * 70)
    print("[4/8] Stage 2 · CART 物理分区 + 分区经验分位数映射")
    print("=" * 70)
    X_train_phys = df.loc[train_idx, PHYS_COLS]
    X_test_phys = df.loc[test_idx, PHYS_COLS]

    # 原实现的树目标 = Stage-1 残差（论文2.5节误写为观测风速，见回复信 R2-9）
    tree_target = obs_train - lgbm_corrected_train
    regime_tree = DecisionTreeRegressor(
        max_depth=CART_MAX_DEPTH, min_samples_leaf=CART_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE)
    regime_tree.fit(X_train_phys, tree_target)

    train_regimes = regime_tree.apply(X_train_phys)
    test_regimes = regime_tree.apply(X_test_phys)
    unique_regimes = np.unique(train_regimes)
    print(f"  分区数 = {len(unique_regimes)}")
    for rg in unique_regimes:
        n_tr = int((train_regimes == rg).sum())
        n_te = int((test_regimes == rg).sum())
        print(f"    叶 {rg}: 训练 {n_tr:,} | 测试 {n_te:,}")

    cascade_test = np.zeros_like(lgbm_corrected_test)
    covered = np.zeros(len(cascade_test), dtype=bool)   # 显式追踪，避免用零值间接推断
    for rg in unique_regimes:
        m_tr = train_regimes == rg
        m_te = test_regimes == rg
        if not np.any(m_te):
            continue
        cascade_test[m_te] = empirical_qm_extrapolate(
            lgbm_corrected_train[m_tr], obs_train[m_tr], lgbm_corrected_test[m_te])
        covered |= m_te

    n_uncovered = int((~covered).sum())
    if n_uncovered > 0:
        # 测试样本落到了训练期未出现过的叶（理论上不应发生），回退为 Stage-1 输出
        print(f"  ⚠️ {n_uncovered} 个测试样本未被任何叶覆盖，已回退为 Stage-1 输出")
        cascade_test[~covered] = lgbm_corrected_test[~covered]
    else:
        print("  ✓ 全部测试样本均被分区覆盖")

    # ⚠️ 训练侧的级联值是「自映射」：用训练集自身的分布映射训练集自身，
    #    其方差比会人为接近 1，**不可用于任何评估或作图**。仅为留存对照而导出，
    #    导出文件中该列已改名为 E_cascade_selfmap_DO_NOT_EVALUATE。
    cascade_train_selfmap = np.zeros_like(lgbm_corrected_train)
    for rg in unique_regimes:
        m_tr = train_regimes == rg
        if not np.any(m_tr):
            continue
        cascade_train_selfmap[m_tr] = empirical_qm_extrapolate(
            lgbm_corrected_train[m_tr], obs_train[m_tr], lgbm_corrected_train[m_tr])

    # Model D: LightGBM + Global QM（复现表4）
    gqm_test = global_qm_extrapolate(lgbm_corrected_train, obs_train, lgbm_corrected_test)

    # Model C: NWP + PC-RQM（复现表4，仅 Stage-2）
    rqm_only_test = fcst_test.astype(float).copy()
    for rg in unique_regimes:
        m_tr = train_regimes == rg
        m_te = test_regimes == rg
        if not np.any(m_te):
            continue
        rqm_only_test[m_te] = empirical_qm_extrapolate(
            fcst_train[m_tr], obs_train[m_tr], fcst_test[m_te])

    # ---------- [4.5/8] 基线训练前的检查点 ----------
    # 说明：核心结果（A–E）此时已全部算完。基线模型（F/G）内存开销大，
    #       一旦崩溃不应导致主结果丢失，因此先把核心预测值落盘。
    print("\n" + "=" * 70)
    print("[4.5/8] 核心结果检查点（基线训练前落盘）")
    print("=" * 70)
    test_meta_ckpt = df.loc[test_idx, [
        "obs_time", "date", "month", "hour", "forecast_lead_time",
        "lead_time_bin", "obs_wind_bin", "init_time"]].copy().reset_index(drop=True)
    test_meta_ckpt["obs"] = obs_test
    test_meta_ckpt["A_raw_nwp"] = fcst_test
    test_meta_ckpt["B_lgbm"] = lgbm_corrected_test
    test_meta_ckpt["C_nwp_pcrqm"] = rqm_only_test
    test_meta_ckpt["D_lgbm_gqm"] = gqm_test
    test_meta_ckpt["E_cascade"] = cascade_test
    test_meta_ckpt["regime_leaf"] = test_regimes
    ckpt_path = os.path.join(OUT_DIR, "predictions_test_coreAE.csv")
    test_meta_ckpt.to_csv(ckpt_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ 核心结果 A–E 已落盘：{ckpt_path}")
    print("    （即使后续基线训练崩溃，S2 仍可基于此文件运行：")
    print("      把 S2 的 PRED_PATH 指向该文件即可，只是缺 F/G/H/I 四列）")

    def report_mem(tag):
        """打印当前进程内存占用。用于监控，非崩溃诊断（本脚本崩溃真因是
        OpenMP 运行时冲突，与内存无关）。"""
        try:
            import psutil
            rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            avail = psutil.virtual_memory().available / 1024 ** 2
            print(f"      [{tag}] 进程占用 {rss:,.0f} MB / 系统可用 {avail:,.0f} MB")
        except ImportError:
            pass

    report_mem("基线训练前")

    # ---------- [5/8] 新增基线（回应 R1-6） ----------
    print("\n" + "=" * 70)
    print("[5/8] 补充基线模型（R1-6）")
    print("=" * 70)

    # F: Random Forest —— 同特征、同残差目标、同切分，**标准配置**
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_samples=RF_MAX_SAMPLES,
        max_features=RF_MAX_FEATURES,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE)
    print(f"  F Random Forest 配置：n_estimators={RF_N_ESTIMATORS}, "
          f"max_depth={RF_MAX_DEPTH}, min_samples_leaf={RF_MIN_SAMPLES_LEAF}, "
          f"max_samples={RF_MAX_SAMPLES}, max_features={RF_MAX_FEATURES}, "
          f"n_jobs={RF_N_JOBS}")
    if RF_MAX_DEPTH is None and RF_MAX_SAMPLES is None:
        print("    → 标准不限深度、全样本 bootstrap 的 Random Forest")
    else:
        print("    ⚠️ 非标准配置：写入回复信时不得表述为「标准 Random Forest」")
    rf.fit(X_train_s, y_train)
    rf_test = np.clip(fcst_test + rf.predict(X_test_s), 0, None)
    print(f"  F Random Forest 训练 {time.time() - t0:.1f}s")
    report_mem("RF 训练后")
    # 立即持久化，随后释放模型对象
    pd.DataFrame({"F_randomforest": rf_test}).to_csv(
        os.path.join(OUT_DIR, "_ckpt_F_randomforest.csv"),
        index=False, encoding="utf-8-sig")
    rf = None
    gc.collect()
    report_mem("RF 释放后")

    # G: HistGradientBoosting —— 第二个梯度提升实现，证明结论不依赖单一库
    #    ⚠️ 放在**独立子进程**中训练：LightGBM 已在本进程加载一套 OpenMP，
    #    再导入 sklearn 的 HGB 会触发第二套 OpenMP 的延迟加载符号解析失败
    #    （0xC06D007F），直接杀死 Jupyter 内核。子进程隔离后，即使失败
    #    也只影响 G 这一列，A–F 与 H/I 全部保留。
    print("\n  G HistGradientBoosting —— 在独立子进程中训练（隔离 OpenMP 冲突）")
    t0 = time.time()
    hgb_test, hgb_n_iter, hgb_err = _train_hgb_in_subprocess(
        X_train_s, y_train, X_test_s, fcst_test, OUT_DIR,
        max_iter=HGB_MAX_ITER, max_leaf_nodes=HGB_MAX_LEAF_NODES,
        random_state=RANDOM_STATE)
    if hgb_test is not None:
        print(f"  G HistGradientBoosting 训练 {time.time() - t0:.1f}s"
              f"（实际迭代 {hgb_n_iter} 轮）")
        pd.DataFrame({"G_histgb": hgb_test}).to_csv(
            os.path.join(OUT_DIR, "_ckpt_G_histgb.csv"),
            index=False, encoding="utf-8-sig")
    else:
        print(f"  ⚠️ G 训练失败：{hgb_err}")
        print("     A–F、H、I 不受影响；G 列将留空（NaN），S2/S4 会自动跳过该模型。")
        print("     处理方式（三选一）：")
        print("       1) 直接忽略 —— R1-6 要求的基线已由 RF/Persistence/线性订正覆盖，")
        print("          第二个梯度提升实现属加分项而非必需；")
        print("       2) 重跑本脚本 —— 子进程隔离后重试不会杀死 Jupyter 内核；")
        print("       3) 解决 OpenMP 冲突后重跑：在 conda 环境中执行")
        print("          conda install -c conda-forge nomkl intel-openmp")
        print("          （或统一为单一 OpenMP 运行时），再重跑本脚本即可补齐 G。")

    # H: Persistence —— 用 t-15min 的观测作为预报（风速预报标准参照）
    #    严格时间匹配：把观测序列整体后移 15 min 再 join，匹配不上记 NaN
    print("  H Persistence (t-15min) ...")
    obs_series = (df[["obs_time", "obs_wind_speed"]]
                  .drop_duplicates(subset=["obs_time"])
                  .sort_values("obs_time")
                  .reset_index(drop=True))
    obs_prev = obs_series.copy()
    obs_prev["obs_time"] = obs_prev["obs_time"] + pd.Timedelta(minutes=15)
    obs_prev = obs_prev.rename(columns={"obs_wind_speed": "persistence"})
    df_test = df.loc[test_idx, ["obs_time"]].merge(obs_prev, on="obs_time", how="left")
    persistence_test = df_test["persistence"].values.astype(float)
    n_nan = int(np.isnan(persistence_test).sum())
    print(f"      覆盖 {len(persistence_test) - n_nan:,} / {len(persistence_test):,}"
          f"（{n_nan:,} 个样本无 t-15min 观测，指标计算时排除并单独报告）")

    # I: Linear bias correction —— 观测 = a×预报 + b，训练集拟合（统计订正代表）
    lin = LinearRegression().fit(fcst_train.reshape(-1, 1), obs_train)
    linear_test = np.clip(lin.predict(fcst_test.reshape(-1, 1)), 0, None)
    print(f"  I Linear bias correction: obs = {lin.coef_[0]:.4f} × fcst "
          f"+ {lin.intercept_:.4f}")

    # ---------- [6/8] 自检：是否复现论文表2/表3 ----------
    print("\n" + "=" * 70)
    print("[6/8] 自检 · 与论文原表比对")
    print("=" * 70)
    got = {
        "Raw NWP": metrics(obs_test, fcst_test),
        "LightGBM": metrics(obs_test, lgbm_corrected_test),
        "Cascade": metrics(obs_test, cascade_test),
    }
    print(f"{'模型':<12}{'指标':<8}{'论文':>10}{'本次':>10}{'差值':>10}  判定")
    print("-" * 56)
    all_ok = True
    for name, d in got.items():
        for k in ["MBE", "RMSE", "MAE", "R2"]:
            paper = PAPER_TABLE2[name][k]
            cur = d[k]
            diff = cur - paper
            ok = abs(diff) < 0.02
            all_ok = all_ok and ok
            print(f"{name:<12}{k:<8}{paper:>10.3f}{cur:>10.3f}{diff:>+10.3f}  "
                  f"{'✅' if ok else '❌ 不一致'}")
    print("-" * 56)
    if all_ok:
        print("✅ 表2 完全复现 —— 本脚本与原实验口径一致，新增基线可直接与之比较。")
    else:
        print("⚠️ 表2 存在差异。可能原因：数据文件版本不同 / lightgbm 版本不同 /")
        print("   early_stopping 行为差异。请先解决一致性，再使用后续基线结果。")

    # 表3 样本量自检
    bins_edges = [0, 5, 10, 15, np.inf]
    got_n = [int(((obs_test >= bins_edges[i]) & (obs_test < bins_edges[i + 1])).sum())
             for i in range(4)]
    print(f"\n表3 样本量自检：本次 {got_n} vs 论文 {PAPER_TABLE3_N}  "
          f"{'✅ 一致' if got_n == PAPER_TABLE3_N else '❌ 不一致'}")

    # ---------- [7/8] 导出预测值 ----------
    print("\n" + "=" * 70)
    print("[7/8] 导出预测值（S2/S3 的唯一输入）")
    print("=" * 70)
    test_meta = df.loc[test_idx, [
        "obs_time", "date", "month", "hour", "forecast_lead_time",
        "lead_time_bin", "obs_wind_bin", "init_time"]].copy()
    if "lat_idx" in df.columns:
        pass  # 不导出坐标，避免脱敏问题
    out = test_meta.reset_index(drop=True)
    out["obs"] = obs_test
    out["A_raw_nwp"] = fcst_test
    out["B_lgbm"] = lgbm_corrected_test
    out["C_nwp_pcrqm"] = rqm_only_test
    out["D_lgbm_gqm"] = gqm_test
    out["E_cascade"] = cascade_test
    out["F_randomforest"] = rf_test
    # G 可能因子进程失败而为 None；此时填 NaN，S2 会自动跳过该模型
    if hgb_test is None:
        out["G_histgb"] = np.nan
    else:
        out["G_histgb"] = hgb_test
    out["H_persistence"] = persistence_test
    out["I_linear"] = linear_test
    out["regime_leaf"] = test_regimes

    # 训练侧预测值（用于 SHAP、CDF 诊断、阈值稳定性）
    tr_meta = df.loc[train_idx, ["obs_time", "date", "month", "forecast_lead_time",
                                 "lead_time_bin"]].copy().reset_index(drop=True)
    tr_out = tr_meta
    tr_out["obs"] = obs_train
    tr_out["A_raw_nwp"] = fcst_train
    tr_out["B_lgbm"] = lgbm_corrected_train
    tr_out["E_cascade_selfmap_DO_NOT_EVALUATE"] = cascade_train_selfmap
    tr_out["regime_leaf"] = train_regimes

    p_test = os.path.join(OUT_DIR, "predictions_test.csv")
    p_train = os.path.join(OUT_DIR, "predictions_train.csv")
    out.to_csv(p_test, index=False, encoding="utf-8-sig")
    tr_out.to_csv(p_train, index=False, encoding="utf-8-sig")
    print(f"  测试集预测值 -> {p_test}  ({len(out):,} 行)")
    print(f"  训练集预测值 -> {p_train}  ({len(tr_out):,} 行)")

    # 保存模型与 scaler，供 S3 做 SHAP 与阈值稳定性
    # 注：rf 已在基线段释放（置 None）以节省内存；hgb 在子进程中训练，
    #     其对象无法跨进程传递。S3 只需要 model_lgb / scaler / regime_tree，
    #     故此处不保存 rf 与 hgb，如需可另行单独持久化。
    import pickle
    pkl = os.path.join(OUT_DIR, "models_and_scaler.pkl")
    with open(pkl, "wb") as f:
        pickle.dump({
            "model_lgb": model_lgb, "scaler": scaler,
            "regime_tree": regime_tree, "lin": lin,
            "feature_cols": FEATURE_COLS, "phys_cols": PHYS_COLS,
            "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            "train_time_lgb": lat_lgb_train,
        }, f)
    print(f"  模型与 scaler -> {pkl}")

    # 保存验证集特征（SHAP 应在 held-out 数据上计算，回应 R2-10）
    val_meta = df.loc[val_idx, ["obs_time", "date", "month"]].copy().reset_index(drop=True)
    val_feat = pd.DataFrame(X_val_s, columns=FEATURE_COLS)
    val_pack = pd.concat([val_meta, val_feat], axis=1)
    val_pack["obs"] = df.loc[val_idx, "obs_wind_speed"].values
    val_pack["fcst"] = df.loc[val_idx, "forecast_wind_speed"].values
    p_val = os.path.join(OUT_DIR, "validation_scaled_features.csv")
    val_pack.to_csv(p_val, index=False, encoding="utf-8-sig")
    print(f"  验证集标准化特征 -> {p_val}  ({len(val_pack):,} 行)")

    # ---------- [8/8] 完成 ----------
    print("\n" + "=" * 70)
    print(f"[8/8] 完成，总耗时 {time.time() - t_start:.1f}s")
    print("=" * 70)
    print("下一步：python S2_metrics_and_tests.py")


if __name__ == "__main__":
    main()
