# -*- coding: utf-8 -*-
"""
Cross-Regional AGB Transfer Learning Framework (Xijiang -> Simao)
迁移学习主代码（仅迁移策略对比）：
1) Direct transfer：源域训练 -> 直接用于目标域预测
2) Fine-tuning：在源域模型基础上，使用目标域训练集继续训练（追加树/增量训练）
3) Retraining：仅用目标域训练集重新训练模型

输入：源域/目标域特征表（csv/xlsx），包含：
  - 目标变量列：Y_COL（如 AGB）
  - 特征列：FEATURE_COLS（可手动指定；若为None则自动推断数值列）

输出：run_时间戳/
  - metrics_summary.csv
  - selected_features.csv（若启用RFE）
  - models/*.joblib（若安装joblib）
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

try:
    import joblib
except Exception:
    joblib = None


# =========================
# 0) 只改这里
# =========================
SOURCE_PATH = r"E:\Xijiang\matched_features.xlsx"   # 源域数据
TARGET_PATH = r"E:\Simao\matched_features.xlsx"    # 目标域数据
OUT_BASE    = r"D:\AGB_transfer_learning"          # 输出目录

Y_COL = "AGB"               # 目标变量列名

FEATURE_COLS = None         # None=自动推断（推荐你先None跑通再固定）
# FEATURE_COLS = ["B2","B3","NDVI","VV","VH","slope","aspect",...]

RANDOM_STATE = 42
TEST_SIZE = 0.30            # 源域/目标域划分比例（建议保持一致）

# 是否启用 RFE
DO_RFE = True
RFE_KEEP = 30
RFE_ESTIMATOR = "RF"        # "RF" or "XGB"

# XGBoost 参数（按你自己的调参结果可替换）
XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# Fine-tuning：在源模型基础上追加的树数
FT_ADD_TREES = 300


# =========================
# 1) 基础函数
# =========================
def read_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"Unsupported file type: {path}")


def infer_feature_cols(df: pd.DataFrame, y_col: str) -> list[str]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in num_cols if c != y_col]
    if not feats:
        raise ValueError("No numeric feature columns found. Please set FEATURE_COLS manually.")
    return feats


def metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return dict(
        R2=float(r2_score(y_true, y_pred)),
        RMSE=float(rmse),
        MAE=float(mean_absolute_error(y_true, y_pred)),
    )


def build_xgb(n_estimators=None) -> xgb.XGBRegressor:
    params = dict(XGB_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    return xgb.XGBRegressor(**params)


def rfe_select(X_train: pd.DataFrame, y_train: np.ndarray, keep: int, estimator_type: str) -> list[str]:
    if keep >= X_train.shape[1]:
        return list(X_train.columns)

    if estimator_type.upper() == "RF":
        est = RandomForestRegressor(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        est = build_xgb()

    selector = RFE(estimator=est, n_features_to_select=int(keep), step=0.1)
    selector.fit(X_train.values, y_train)
    return [c for c, m in zip(X_train.columns, selector.support_) if m]


# =========================
# 2) 主流程
# =========================
def main():
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(OUT_BASE) / run_tag
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # --- 读数据 ---
    df_s = read_table(SOURCE_PATH)
    df_t = read_table(TARGET_PATH)

    # --- 特征列 ---
    feats = FEATURE_COLS if FEATURE_COLS else infer_feature_cols(df_s, Y_COL)
    missing = [c for c in feats if c not in df_t.columns]
    if missing:
        raise ValueError(f"Target domain missing feature columns: {missing[:10]} ... (total {len(missing)})")

    # --- 取y并清理 ---
    y_s = pd.to_numeric(df_s[Y_COL], errors="coerce").to_numpy(float)
    y_t = pd.to_numeric(df_t[Y_COL], errors="coerce").to_numpy(float)

    # --- 划分源域/目标域 ---
    idx_s = np.arange(len(df_s))
    idx_t = np.arange(len(df_t))

    s_tr, s_te = train_test_split(idx_s, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    t_tr, t_te = train_test_split(idx_t, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    Xs_tr_raw = df_s.loc[s_tr, feats]
    Xs_te_raw = df_s.loc[s_te, feats]
    Xt_tr_raw = df_t.loc[t_tr, feats]
    Xt_te_raw = df_t.loc[t_te, feats]

    ys_tr = y_s[s_tr]
    ys_te = y_s[s_te]
    yt_tr = y_t[t_tr]
    yt_te = y_t[t_te]

    # --- 缺失值处理：用源域训练集拟合imputer（统一pipeline） ---
    imputer = SimpleImputer(strategy="median")
    imputer.fit(Xs_tr_raw.values)

    def imp(dfX: pd.DataFrame) -> pd.DataFrame:
        arr = imputer.transform(dfX.values)
        return pd.DataFrame(arr, columns=dfX.columns, index=dfX.index)

    Xs_tr = imp(Xs_tr_raw)
    Xs_te = imp(Xs_te_raw)
    Xt_tr = imp(Xt_tr_raw)
    Xt_te = imp(Xt_te_raw)

    # --- RFE（可选）：只在源域训练集做筛选，应用到目标域 ---
    selected_feats = list(feats)
    if DO_RFE:
        selected_feats = rfe_select(Xs_tr, ys_tr, RFE_KEEP, RFE_ESTIMATOR)
        pd.DataFrame({"selected_features": selected_feats}).to_csv(
            out_dir / "selected_features.csv", index=False, encoding="utf-8-sig"
        )
        Xs_tr = Xs_tr[selected_feats]
        Xs_te = Xs_te[selected_feats]
        Xt_tr = Xt_tr[selected_feats]
        Xt_te = Xt_te[selected_feats]

    # ==========================================================
    # Strategy 1: Source training (baseline)
    # ==========================================================
    model_source = build_xgb()
    model_source.fit(Xs_tr.values, ys_tr)

    pred_s = model_source.predict(Xs_te.values)
    m_source = metrics(ys_te, pred_s)

    if joblib:
        joblib.dump(model_source, out_dir / "models" / "xgb_source.joblib")

    # ==========================================================
    # Strategy 2: Direct transfer (source -> target)
    # ==========================================================
    pred_direct = model_source.predict(Xt_te.values)
    m_direct = metrics(yt_te, pred_direct)

    # ==========================================================
    # Strategy 3: Fine-tuning (continue training on target)
    #   在 source 模型 booster 基础上追加树，在目标域训练集继续训练
    # ==========================================================
    model_ft = build_xgb(n_estimators=XGB_PARAMS["n_estimators"] + FT_ADD_TREES)
    model_ft.fit(Xt_tr.values, yt_tr, xgb_model=model_source.get_booster())

    pred_ft = model_ft.predict(Xt_te.values)
    m_ft = metrics(yt_te, pred_ft)

    if joblib:
        joblib.dump(model_ft, out_dir / "models" / "xgb_finetune.joblib")

    # ==========================================================
    # Strategy 4: Retraining (train from scratch on target only)
    # ==========================================================
    model_rt = build_xgb()
    model_rt.fit(Xt_tr.values, yt_tr)

    pred_rt = model_rt.predict(Xt_te.values)
    m_rt = metrics(yt_te, pred_rt)

    if joblib:
        joblib.dump(model_rt, out_dir / "models" / "xgb_retrain.joblib")

    # --- 汇总输出 ---
    summary = pd.DataFrame([
        {"Setting": "Source (Xijiang) baseline", **m_source},
        {"Setting": "Target direct transfer", **m_direct},
        {"Setting": "Target fine-tuning", **m_ft},
        {"Setting": "Target retraining", **m_rt},
    ])
    summary.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Transfer learning results ===")
    print(summary.to_string(index=False))
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
