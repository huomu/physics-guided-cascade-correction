# %% [markdown]
# # S1 · 复现原实验 + 补充基线 + 导出预测值（分 Cell 版）
#
# **本文件按你已跑通的三个 Cell 原样保留**，仅在末尾补两个 Cell：
# - **Cell 4**：导出 `validation_scaled_features.csv` —— **S3 的 SHAP 重算必需**，
#   你的 Cell 3 只存了模型与 scaler，缺这个文件 S3 会直接跳过 SHAP。
# - **Cell 5**：交接检查，确认 S2 / S3 / S4 的前置文件是否齐全。
#
# ## 运行方式
# - VS Code：直接打开本 `.py`，每个 `# %%` 上方会出现 **Run Cell**，
#   右上角选择你现有的 Jupyter 内核（`py311`）后逐格运行。
# - 或把每个 `# %%` 之间的代码块复制进你的 `compare_reveiwer.ipynb` 对应 Cell。
#
# ## Cell 依赖顺序（不可跳格）
# ```
# Cell 1  配置/导入/通用函数
#   └─> Cell 2  数据加载 → 切分 → LightGBM → CART → QM → 核心 A–E 落盘
#         └─> Cell 3  F/G/H/I 基线 + 表2/表3 自检 + 最终导出
#               └─> Cell 4  补充导出（S3 依赖）
#                     └─> Cell 5  交接检查
# ```
#
# ## 关于内核崩溃（0xC06D007F）的说明 —— 请勿"优化"掉子进程
# 崩溃真因**不是内存**（日志显示进程仅 472 MB、系统可用 20 GB），而是
# LightGBM 与 scikit-learn 的 HistGradientBoosting 各自携带一套 OpenMP 运行时，
# 同进程先后加载时 Windows 延迟加载找不到 DLL 导出符号，进程静默终止。
#
# **真正起作用的是「进程隔离」**：`_train_hgb_in_subprocess` 启动一个
# 全新的 Python 子进程，该子进程**只导入 numpy + sklearn、从不导入 lightgbm**，
# 因此根本不存在两套 OpenMP 共存的问题。
#
# Cell 1 里的 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", ...)` 写在
# `import lightgbm` **之后**，对主进程实际上已来不及生效——它只是子进程的
# 双保险。所以：
# - 不要把 G 改回主进程内直接 `HistGradientBoostingRegressor(...).fit(...)`；
# - 也不要因为"env 变量看起来没用"就删掉子进程逻辑。

# %%
# ============================================================================
# Cell 1 · 配置、导入、通用函数
# ============================================================================
import os
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
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

# 注：这些环境变量写在 import 之后，对主进程已来不及生效（DLL 已加载）。
# 真正避免 OpenMP 冲突的是下面 _train_hgb_in_subprocess 的进程隔离；
# 此处保留作为子进程侧的双保险，删除无妨但无益。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

warnings.filterwarnings("ignore")


def _first_existing(candidates, env_key=None, must_exist=True):
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


_CSV_CANDIDATES = [
    r"D:\data\DeChang\analysis_results_06_10\paired_data.csv",
    os.path.join(os.getcwd(), "paired_data.csv"),
]
_OUT_CANDIDATES = [
    r"D:\data\DeChang\analysis_results_06_10\补充实验产出",
    os.path.join(os.getcwd(), "补充实验产出"),
]

CSV_PATH = _first_existing(_CSV_CANDIDATES, env_key="DECHANG_CSV", must_exist=True)
OUT_DIR = _first_existing(_OUT_CANDIDATES, env_key="DECHANG_OUT", must_exist=False)
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_QM_POINTS = 1000
CART_MAX_DEPTH = 3
CART_MIN_SAMPLES_LEAF = 500   # 与原实现一致；论文 2.5 节写 1000，需在回复信中说明

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

PAPER_TABLE2 = {
    "Raw NWP": {"MBE": 1.803, "RMSE": 4.165, "MAE": 3.207, "R2": -1.843},
    "LightGBM": {"MBE": 0.042, "RMSE": 2.076, "MAE": 1.618, "R2": 0.294},
    "Cascade": {"MBE": 0.045, "RMSE": 2.432, "MAE": 1.898, "R2": 0.030},
}
PAPER_TABLE3_N = [37055, 27661, 2191, 13]


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
    """与原实现一致的经验分位数映射（线性外推版）。"""
    src_train = np.asarray(src_train, float)
    tgt_train = np.asarray(tgt_train, float)
    src_test = np.asarray(src_test, float)
    if src_test.size == 0 or src_train.size < 10:
        return src_test.copy()
    qs = np.linspace(0, 100, n_points)
    src_q = np.percentile(src_train, qs)
    tgt_q = np.percentile(tgt_train, qs)
    # np.unique 保证 interp1d 的 x 严格递增（分位点重复时原实现会报错或非单调）
    src_q, uidx = np.unique(src_q, return_index=True)
    tgt_q = tgt_q[uidx]
    if src_q.size < 2:
        return src_test.copy()
    mapping = interp1d(src_q, tgt_q, kind="linear", bounds_error=False,
                       fill_value="extrapolate")
    return np.clip(mapping(src_test), 0, None)


def global_qm_extrapolate(src_train, tgt_train, src_test):
    return empirical_qm_extrapolate(src_train, tgt_train, src_test)


# ---- G 基线的子进程隔离训练 ----
# 子进程源码：只导入 numpy + sklearn，从不导入 lightgbm，
# 因此不存在两套 OpenMP 运行时共存 → 不会触发 0xC06D007F。
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
    max_iter=max_iter,
    learning_rate=0.05,
    max_depth=7,
    max_leaf_nodes=max_leaf,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=seed,
)
model.fit(X_tr, y_tr)
pred = np.clip(fcst_te + model.predict(X_te), 0, None)
np.savez(os.path.join(work, "hgb_output.npz"), pred=pred, n_iter=int(model.n_iter_))
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
        np.savez(
            os.path.join(work, "hgb_input.npz"),
            X_train=np.asarray(X_train_s, dtype=np.float64),
            y_train=np.asarray(y_train, dtype=np.float64),
            X_test=np.asarray(X_test_s, dtype=np.float64),
            fcst_test=np.asarray(fcst_test, dtype=np.float64),
        )

        child = os.path.join(work, "hgb_child.py")
        with open(child, "w", encoding="utf-8") as f:
            f.write(_HGB_CHILD_SOURCE)

        proc = subprocess.run(
            [sys.executable, child, work, str(max_iter),
             str(max_leaf_nodes), str(random_state)],
            capture_output=True, text=True, timeout=timeout,
        )

        out_path = os.path.join(work, "hgb_output.npz")
        if proc.returncode != 0 or not os.path.exists(out_path):
            err = (proc.stderr or proc.stdout or "").strip()
            code = proc.returncode
            hint = ""
            # 0xC06D007F == -1073740793（有符号）== 3228369023（无符号）
            if code in (-1073740793, 3228369023) or "C06D007F" in err.upper():
                hint = ("  [0xC06D007F: OpenMP 运行时冲突 —— 子进程已隔离仍失败，"
                        "建议 conda install -c conda-forge nomkl intel-openmp]")
            return None, None, f"subprocess exit code {code}{hint}\n{err[-1500:]}"

        d = np.load(out_path)
        return d["pred"], int(d["n_iter"]), None
    except subprocess.TimeoutExpired:
        return None, None, f"subprocess timeout (> {timeout}s)"
    except Exception as exc:                      # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


print("[config] CSV_PATH =", CSV_PATH)
print("[config] OUT_DIR   =", OUT_DIR)
print("[config] ready")

# %%
# ============================================================================
# Cell 2 · 数据加载、切分、LightGBM、CART、QM、核心 A–E 落盘
# ============================================================================
t_start = time.time()

df = pd.read_csv(CSV_PATH)
df["obs_time"] = pd.to_datetime(df["obs_time"])
df = df.sort_values("obs_time").reset_index(drop=True)

print("df shape:", df.shape)
print("nulls:", int(df.isna().sum().sum()))

missing = [c for c in FEATURE_COLS + PHYS_COLS + ["obs_wind_speed", "forecast_wind_speed"]
           if c not in df.columns]
if missing:
    raise KeyError(f"missing columns: {missing}")

df["wind_speed_error"] = df["obs_wind_speed"] - df["forecast_wind_speed"]
df["year_month"] = df["obs_time"].dt.to_period("M")
df["date"] = df["obs_time"].dt.date

# ---- 复现原实现的「每月内部 70/15/15」切分 ----
train_idx, val_idx, test_idx = [], [], []
for ym, group in df.groupby("year_month"):
    n = len(group)
    tr_end = int(n * 0.70)
    va_end = int(n * 0.85)
    idx = group.index.values
    train_idx.extend(idx[:tr_end])
    val_idx.extend(idx[tr_end:va_end])
    test_idx.extend(idx[va_end:])

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

print("train/val/test:", len(train_idx), len(val_idx), len(test_idx))

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

# ---- Stage 1: LightGBM 残差订正（原参数；早停用验证集，不用测试集）----
model_lgb = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    objective="regression",
)
model_lgb.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

lgbm_corrected_train = np.clip(fcst_train + model_lgb.predict(X_train_s), 0, None)
lgbm_corrected_test = np.clip(fcst_test + model_lgb.predict(X_test_s), 0, None)

# ---- Stage 2: CART 物理分区 ----
# 注意：树目标是 Stage-1 残差（obs − V_ml），不是论文 2.5 节所写的「观测风速」。
# 这一点须在回复信 R2-9 中如实更正（对作者有利，见应对策略文档）。
X_train_phys = df.loc[train_idx, PHYS_COLS]
X_test_phys = df.loc[test_idx, PHYS_COLS]
tree_target = obs_train - lgbm_corrected_train

regime_tree = DecisionTreeRegressor(
    max_depth=CART_MAX_DEPTH,
    min_samples_leaf=CART_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
)
regime_tree.fit(X_train_phys, tree_target)

train_regimes = regime_tree.apply(X_train_phys)
test_regimes = regime_tree.apply(X_test_phys)
unique_regimes = np.unique(train_regimes)

print("分区数 =", len(unique_regimes))
for rg in unique_regimes:
    print(f"  叶 {rg}: 训练 {int((train_regimes == rg).sum()):,} | "
          f"测试 {int((test_regimes == rg).sum()):,}")

# ---- E: Cascade = LightGBM + PC-RQM ----
# CDF 只用训练期数据估计；测试样本仅通过 apply() 路由 → 无 target leakage。
cascade_test = np.zeros_like(lgbm_corrected_test)
covered = np.zeros(len(cascade_test), dtype=bool)
for rg in unique_regimes:
    m_tr = train_regimes == rg
    m_te = test_regimes == rg
    if not np.any(m_te):
        continue
    cascade_test[m_te] = empirical_qm_extrapolate(
        lgbm_corrected_train[m_tr],
        obs_train[m_tr],
        lgbm_corrected_test[m_te],
    )
    covered |= m_te

n_uncovered = int((~covered).sum())
if n_uncovered > 0:
    cascade_test[~covered] = lgbm_corrected_test[~covered]
    print(f"⚠️ {n_uncovered} 个测试样本未被分区覆盖，已回退为 Stage-1 输出")
else:
    print("✓ 全部测试样本均被分区覆盖")
print("trained LightGBM and CART; cascade finite:", bool(np.isfinite(cascade_test).all()))

# ---- D: LightGBM + Global QM ----
gqm_test = global_qm_extrapolate(lgbm_corrected_train, obs_train, lgbm_corrected_test)

# ---- C: NWP + PC-RQM（仅 Stage 2）----
rqm_only_test = fcst_test.astype(float).copy()
for rg in unique_regimes:
    m_tr = train_regimes == rg
    m_te = test_regimes == rg
    if not np.any(m_te):
        continue
    rqm_only_test[m_te] = empirical_qm_extrapolate(
        fcst_train[m_tr],
        obs_train[m_tr],
        fcst_test[m_te],
    )

# ---- 核心结果 A–E 检查点：基线开训前先落盘，崩溃也不丢主结果 ----
test_meta_ckpt = df.loc[test_idx, [
    "obs_time", "date", "month", "hour", "forecast_lead_time",
    "lead_time_bin", "obs_wind_bin", "init_time"
]].copy().reset_index(drop=True)
test_meta_ckpt["obs"] = obs_test
test_meta_ckpt["A_raw_nwp"] = fcst_test
test_meta_ckpt["B_lgbm"] = lgbm_corrected_test
test_meta_ckpt["C_nwp_pcrqm"] = rqm_only_test
test_meta_ckpt["D_lgbm_gqm"] = gqm_test
test_meta_ckpt["E_cascade"] = cascade_test
test_meta_ckpt["regime_leaf"] = test_regimes

core_path = os.path.join(OUT_DIR, "predictions_test_coreAE.csv")
test_meta_ckpt.to_csv(core_path, index=False, encoding="utf-8-sig")
print("core A-E saved:", core_path)

print("Cell 2 finished; proceed to Cell 3")

# %%
# ============================================================================
# Cell 3 · F/G/H/I 基线 + 表2/表3 自检 + 最终导出
# ============================================================================
# RF 采用**标准配置**（不限深度、全样本 bootstrap）。
# 此前"RF 会占用数百 GB 内存"的说法是错误归因 —— 崩溃真因是 OpenMP 冲突。
# ⚠️ 若把下面任何一项改成非默认值，回复信中不得表述为"标准 Random Forest"。
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1
RF_MAX_SAMPLES = None
RF_MAX_FEATURES = 1.0
RF_N_JOBS = 4
HGB_MAX_ITER = 300
HGB_MAX_LEAF_NODES = 31

# ---------- F: Random Forest ----------
rf = RandomForestRegressor(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    max_samples=RF_MAX_SAMPLES,
    max_features=RF_MAX_FEATURES,
    n_jobs=RF_N_JOBS,
    random_state=RANDOM_STATE,
)
rf.fit(X_train_s, y_train)
rf_test = np.clip(fcst_test + rf.predict(X_test_s), 0, None)
print("F_RF_OK:", rf_test.shape, np.isfinite(rf_test).all())

# ---------- G: HistGradientBoosting（独立子进程，隔离 OpenMP 冲突）----------
hgb_test, hgb_n_iter, hgb_err = _train_hgb_in_subprocess(
    X_train_s,
    y_train,
    X_test_s,
    fcst_test,
    OUT_DIR,
    max_iter=HGB_MAX_ITER,
    max_leaf_nodes=HGB_MAX_LEAF_NODES,
    random_state=RANDOM_STATE,
)
if hgb_test is not None:
    print("G_HGB_OK:", hgb_test.shape, np.isfinite(hgb_test).all(), "n_iter=", hgb_n_iter)
else:
    print("G_HGB_FAILED:", hgb_err)
    print("  → A–F、H、I 不受影响；G 列填 NaN，S2/S4 会自动跳过该模型。")
    print("  → R1-6 要求的基线已由 RF + Persistence + 线性订正覆盖，G 属加分项。")

# ---------- H: Persistence（t−15min 观测作为预报）----------
obs_series = (
    df[["obs_time", "obs_wind_speed"]]
    .drop_duplicates(subset=["obs_time"])
    .sort_values("obs_time")
    .reset_index(drop=True)
)
obs_prev = obs_series.copy()
obs_prev["obs_time"] = obs_prev["obs_time"] + pd.Timedelta(minutes=15)
obs_prev = obs_prev.rename(columns={"obs_wind_speed": "persistence"})

# 左表 obs_time 有重复（同一时刻被多个起报周期配对），右表已去重 →
# left merge 不会扩张行数，persistence_test 与 test_idx 顺序严格对齐。
df_test = df.loc[test_idx, ["obs_time"]].merge(obs_prev, on="obs_time", how="left")
persistence_test = df_test["persistence"].values.astype(float)
print("H_PERSISTENCE_COVERED:", int(np.isfinite(persistence_test).sum()), "/",
      len(persistence_test))

# ---------- I: Linear bias correction ----------
lin = LinearRegression().fit(fcst_train.reshape(-1, 1), obs_train)
linear_test = np.clip(lin.predict(fcst_test.reshape(-1, 1)), 0, None)
print("I_LINEAR_OK:", linear_test.shape, np.isfinite(linear_test).all(),
      "coef=", lin.coef_[0], "intercept=", lin.intercept_)

# ---------- 自检：是否复现论文表 2 / 表 3 ----------
eval_map = {
    "Raw NWP": metrics(obs_test, fcst_test),
    "LightGBM": metrics(obs_test, lgbm_corrected_test),
    "Cascade": metrics(obs_test, cascade_test),
}

print("\n[表2 自检]")
table2_all_ok = True
for name, stats in eval_map.items():
    for metric in ["MBE", "RMSE", "MAE", "R2"]:
        paper = PAPER_TABLE2[name][metric]
        cur = stats[metric]
        diff = cur - paper
        ok = abs(diff) < 0.02
        table2_all_ok = table2_all_ok and ok
        print(f"{name:<8} {metric:<4} paper={paper:>8.3f} cur={cur:>8.3f} "
              f"diff={diff:>+8.3f} {'OK' if ok else 'DIFF'}")

bins_edges = [0, 5, 10, 15, np.inf]
table3 = [int(((obs_test >= bins_edges[i]) & (obs_test < bins_edges[i + 1])).sum())
          for i in range(4)]
table3_ok = (table3 == PAPER_TABLE3_N)
print("\n[表3 自检] 本次:", table3, "论文:", PAPER_TABLE3_N,
      "->", "OK" if table3_ok else "DIFF")

if table2_all_ok and table3_ok:
    print("\n✅ 表2 与表3 完全复现 —— 本脚本与原实验口径一致，"
          "新增基线 F/G/H/I 可与 A–E 直接比较。")
else:
    print("\n⚠️ 自检存在差异。可能原因：数据文件版本不同 / lightgbm 版本不同 / "
          "early_stopping 行为差异。")
    print("   请先解决一致性，再使用 F/G/H/I 的对比结果与 S2 的显著性检验。")

# ---------- 最终导出 ----------
test_meta = df.loc[test_idx, [
    "obs_time", "date", "month", "hour", "forecast_lead_time",
    "lead_time_bin", "obs_wind_bin", "init_time"
]].copy().reset_index(drop=True)

out = test_meta
out["obs"] = obs_test
out["A_raw_nwp"] = fcst_test
out["B_lgbm"] = lgbm_corrected_test
out["C_nwp_pcrqm"] = rqm_only_test
out["D_lgbm_gqm"] = gqm_test
out["E_cascade"] = cascade_test
out["F_randomforest"] = rf_test
out["G_histgb"] = np.nan if hgb_test is None else hgb_test
out["H_persistence"] = persistence_test
out["I_linear"] = linear_test
out["regime_leaf"] = test_regimes

p_test = os.path.join(OUT_DIR, "predictions_test.csv")
out.to_csv(p_test, index=False, encoding="utf-8-sig")
print("\nEXPORT_OK:", p_test)

# ---------- 保存模型 bundle（供 S3 做 SHAP 与阈值稳定性）----------
# 注：不保存 rf（对象巨大且 S3 不需要）；hgb 在子进程中训练，对象无法跨进程传递。
import pickle
pkl = os.path.join(OUT_DIR, "models_and_scaler.pkl")
with open(pkl, "wb") as f:
    pickle.dump({
        "model_lgb": model_lgb,
        "scaler": scaler,
        "regime_tree": regime_tree,
        "lin": lin,
        "feature_cols": FEATURE_COLS,
        "phys_cols": PHYS_COLS,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "best_iteration": getattr(model_lgb, "best_iteration_", None),
    }, f)
print("MODEL_BUNDLE_OK:", pkl)

print("\n[done] Cell 3 finished; proceed to Cell 4")

# %%
# ============================================================================
# Cell 4 · 补充导出（S3 必需）
# ----------------------------------------------------------------------------
# ⚠️ 你原来的 Cell 3 缺这一步：S3 的 SHAP 重算要读
#    validation_scaled_features.csv，缺文件时 S3 会打印
#    "❌ 缺少 models_and_scaler.pkl 或 validation_scaled_features.csv" 并跳过整节。
#
# 为什么必须用**验证集**而不是训练集算 SHAP（回应 R2-10）：
#   原图 7 是在训练集上算的，且解释对象是另训的 100 棵简化模型。
#   对已拟合的模型在训练集上算 SHAP 会放大特征贡献、失真排序。
#   这里导出 held-out 验证集的标准化特征，配合 bundle 里的正式 Stage-1 模型，
#   S3 才能给出可写进论文的 SHAP 结果。
# ============================================================================

# ---- 4.1 验证集标准化特征（S3 必需）----
val_meta = df.loc[val_idx, ["obs_time", "date", "month"]].copy().reset_index(drop=True)
val_feat = pd.DataFrame(X_val_s, columns=FEATURE_COLS)
val_pack = pd.concat([val_meta, val_feat], axis=1)
val_pack["obs"] = df.loc[val_idx, "obs_wind_speed"].values
val_pack["fcst"] = df.loc[val_idx, "forecast_wind_speed"].values

val_path = os.path.join(OUT_DIR, "validation_scaled_features.csv")
val_pack.to_csv(val_path, index=False, encoding="utf-8-sig")
print("VAL_FEATURES_OK:", val_path, f"({len(val_pack):,} 行 × {val_pack.shape[1]} 列)")

# ---- 4.2 训练侧预测值（可选，默认关闭）----
# 用途：量化「F_ML 由训练集自身预测值估计 → 分布人为偏窄 → 映射过度拉伸尾部」
#       这一自映射偏差，可作为回应 R2-1 的主动加分项（见应对策略文档）。
# 代价：需在训练集上重跑一遍分区 QM，约数十秒。
# ⚠️ 其中 E_cascade_selfmap 是「用训练集分布映射训练集自身」，方差比会人为
#    接近 1，**严禁用于任何评估、作图或写入论文**，仅供偏差诊断对照。
EXPORT_TRAIN_PREDICTIONS = False

if EXPORT_TRAIN_PREDICTIONS:
    cascade_train_selfmap = np.zeros_like(lgbm_corrected_train)
    for rg in unique_regimes:
        m_tr = train_regimes == rg
        if not np.any(m_tr):
            continue
        cascade_train_selfmap[m_tr] = empirical_qm_extrapolate(
            lgbm_corrected_train[m_tr], obs_train[m_tr], lgbm_corrected_train[m_tr])

    tr_meta = df.loc[train_idx, ["obs_time", "date", "month",
                                 "forecast_lead_time", "lead_time_bin"]
                     ].copy().reset_index(drop=True)
    tr_meta["obs"] = obs_train
    tr_meta["A_raw_nwp"] = fcst_train
    tr_meta["B_lgbm"] = lgbm_corrected_train
    tr_meta["E_cascade_selfmap_DO_NOT_EVALUATE"] = cascade_train_selfmap
    tr_meta["regime_leaf"] = train_regimes

    p_train = os.path.join(OUT_DIR, "predictions_train.csv")
    tr_meta.to_csv(p_train, index=False, encoding="utf-8-sig")
    print("TRAIN_PRED_OK:", p_train, f"({len(tr_meta):,} 行)")
    print("  ⚠️ E_cascade_selfmap_DO_NOT_EVALUATE 为自映射结果，不可用于评估")
else:
    print("（已跳过训练侧预测值导出；如需做 R2-1 的自映射偏差诊断，"
          "把 EXPORT_TRAIN_PREDICTIONS 改为 True 后重跑本 Cell）")

# ---- 4.3 释放不再需要的大对象，降低后续 Cell 的内存压力 ----
rf = None
gc.collect()
print("\n[done] Cell 4 finished; proceed to Cell 5")

# %%
# ============================================================================
# Cell 5 · 交接检查：确认 S2 / S3 / S4 的前置文件齐全
# ----------------------------------------------------------------------------
# 跑完本 Cell 再去做后续脚本，避免白跑。
# ============================================================================
REQUIRED = {
    "predictions_test.csv":            ["S2", "S4"],
    "predictions_test_coreAE.csv":     ["S2/S4 的回退输入（基线崩溃时使用）"],
    "models_and_scaler.pkl":           ["S3", "S4"],
    "validation_scaled_features.csv":  ["S3（SHAP 重算）"],
}

print("=" * 72)
print("交接检查 · 输出目录：", OUT_DIR)
print("=" * 72)
all_present = True
for fname, consumers in REQUIRED.items():
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / 1024 ** 2
        print(f"  ✅ {fname:<38} {size_mb:>8.2f} MB   <- {', '.join(consumers)}")
    else:
        all_present = False
        print(f"  ❌ {fname:<38} {'缺失':>10}   <- {', '.join(consumers)}")

print("-" * 72)

# 检查 predictions_test.csv 的列完整性（决定 S2 能覆盖哪些模型）
p_full = os.path.join(OUT_DIR, "predictions_test.csv")
if os.path.exists(p_full):
    cols = pd.read_csv(p_full, nrows=0).columns.tolist()
    expected_models = ["A_raw_nwp", "B_lgbm", "C_nwp_pcrqm", "D_lgbm_gqm",
                       "E_cascade", "F_randomforest", "G_histgb",
                       "H_persistence", "I_linear"]
    print("模型列检查：")
    for c in expected_models:
        if c in cols:
            s = pd.read_csv(p_full, usecols=[c])[c]
            n_nan = int(s.isna().sum())
            flag = "✅" if n_nan == 0 else f"⚠️ 全为空({n_nan:,})" if n_nan == len(s) else f"⚠️ 缺失 {n_nan:,}"
            print(f"  {flag}  {c}")
        else:
            print(f"  ❌ 列不存在  {c}")
    print("-" * 72)

if all_present:
    print("✅ S1 全部产出就绪。下一步依次运行：")
    print(f"     python S2_metrics_and_tests.py     # 分布指标 + 显著性 + 事件类 + 分时效")
    print(f"     python S3_shap_stability_latency.py # SHAP 重算 + CART 阈值稳定性 + 时延")
    print(f"     python S4_figures.py                # 出版级图件重绘（600 dpi TIFF）")
else:
    print("⚠️ 存在缺失文件，请先补齐再运行后续脚本。")

print("\n提示：S2/S3/S4 的输出目录使用同一套候选顺序探测，")
print("      若它们报「找不到文件」，设环境变量 DECHANG_OUT 指向本目录即可：")
print(f"        os.environ['DECHANG_OUT'] = r'{OUT_DIR}'")
