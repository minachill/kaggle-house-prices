import copy

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor



def run_cv(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 123,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    KFold CV を実行する。LightGBM / XGBoost / CatBoost に対応。

    Parameters
    ----------
    model : sklearn-compatible estimator
        LGBMRegressor, XGBRegressor, CatBoostRegressor など。
        各 fold で deepcopy して使用する。

    Returns
    -------
    metrics : np.ndarray, shape (n_splits, 3)
        各 fold の [fold, rmse_train, rmse_val]
    imp : pd.DataFrame
        特徴量重要度 (fold 平均)
    """
    is_lgbm = isinstance(model, lgb.LGBMRegressor)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = []
    imp = pd.DataFrame()

    for nfold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        x_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        x_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        fold_model = copy.deepcopy(model)

        if is_lgbm:
            fold_model.fit(
                x_tr, y_tr,
                eval_set=[(x_tr, y_tr), (x_va, y_va)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(0),
                ],
            )
        elif isinstance(fold_model, xgb.XGBRegressor):
            fold_model.set_params(early_stopping_rounds=100)
            fold_model.fit(
                x_tr, y_tr,
                eval_set=[(x_va, y_va)],
                verbose=False,
            )
        elif isinstance(fold_model, CatBoostRegressor):
            cat_features = [col for col in x_tr.columns if x_tr[col].dtype.name == "category"]
            fold_model.fit(
                x_tr, y_tr,
                cat_features=cat_features,
                eval_set=(x_va, y_va),
                verbose=False,
            )
        else:
            fold_model.fit(x_tr, y_tr)

        rmse_tr = rmse(y_tr, fold_model.predict(x_tr))
        rmse_va = rmse(y_va, fold_model.predict(x_va))
        print(f"[fold {nfold}] tr: {rmse_tr:.5f}, va: {rmse_va:.5f}")
        metrics.append([nfold, rmse_tr, rmse_va])

        if hasattr(fold_model, "feature_importances_"):
            imp_values = fold_model.feature_importances_
        elif hasattr(fold_model, "coef_"):
            imp_values = np.abs(fold_model.coef_)
        else:
            imp_values = np.zeros(X_train.shape[1])

        _imp = pd.DataFrame({
            "col": X_train.columns,
            "imp": imp_values,
            "nfold": nfold,
        })
        imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

    metrics = np.array(metrics)
    print("[cv] tr: {:.5f}±{:.5f}, va: {:.5f}±{:.5f}".format(
        metrics[:, 1].mean(), metrics[:, 1].std(),
        metrics[:, 2].mean(), metrics[:, 2].std(),
    ))
    imp = imp.groupby("col")["imp"].agg(["mean", "std"])
    imp.columns = ["imp", "imp_std"]
    return metrics, imp.sort_values("imp", ascending=False)


def rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_submission(
    model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    filepath: str,
) -> pd.DataFrame:
    """予測値を log1p 逆変換して submission CSV を保存する"""
    y_pred = np.expm1(model.predict(X_test))
    df_submit = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred})
    df_submit.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")
    return df_submit
