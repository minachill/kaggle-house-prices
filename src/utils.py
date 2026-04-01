import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb


def run_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    n_splits: int = 5,
    random_state: int = 123,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    LightGBM の KFold CV を実行する。

    Returns
    -------
    metrics : np.ndarray, shape (n_splits, 3)
        各 fold の [fold, rmse_train, rmse_val]
    imp : pd.DataFrame
        特徴量重要度 (fold 平均)
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = []
    imp = pd.DataFrame()

    for nfold, (train_idx, val_idx) in enumerate(cv.split(X_train)):
        x_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        x_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_tr, y_tr), (x_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(0),
            ],
        )

        rmse_tr = rmse(y_tr, model.predict(x_tr))
        rmse_va = rmse(y_va, model.predict(x_va))
        print(f"[fold {nfold}] tr: {rmse_tr:.5f}, va: {rmse_va:.5f}")
        metrics.append([nfold, rmse_tr, rmse_va])

        _imp = pd.DataFrame({
            "col": X_train.columns,
            "imp": model.feature_importances_,
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
