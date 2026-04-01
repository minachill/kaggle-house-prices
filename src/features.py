import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


QUAL_MAP = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

GARAGE_FINISH_MAP = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}

FUNCTIONAL_MAP = {
    "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
    "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """数値・組み合わせ特徴量の生成（train/test 結合済み df に適用）"""
    df = df.copy()

    # 面積
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"]
        + df["3SsnPorch"] + df["ScreenPorch"]
    )
    df["TotalSF_with_porch"] = df["TotalSF"] + df["TotalPorchSF"]
    df["TotalSF_log"] = np.log1p(df["TotalSF"])
    df["GrLivArea_log"] = np.log1p(df["GrLivArea"])

    # 風呂
    df["TotalBath"] = (
        df["FullBath"]
        + df["HalfBath"] * 0.5
        + df["BsmtFullBath"]
        + df["BsmtHalfBath"] * 0.5
    )

    # 品質
    df["KitchenQual_num"] = df["KitchenQual"].map(QUAL_MAP).astype(float)
    df["BsmtQual_num"] = df["BsmtQual"].map(QUAL_MAP)
    df["TotalQual"] = df["OverallQual"] + df["KitchenQual_num"] + df["BsmtQual_num"]
    df["TotalQual_SF"] = df["TotalQual"] * df["TotalSF"]
    df["TotalQual_SF_log"] = np.log1p(df["TotalQual_SF"])
    df["Qual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["Qual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["Qual_Kitchen"] = df["OverallQual"] * df["KitchenQual_num"]
    df["KitchenScore"] = df["KitchenQual_num"] / df["KitchenAbvGr"]

    # 年数
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["Qual_Age"] = df["OverallQual"] * df["HouseAge"]
    df["Qual_Remod"] = df["OverallQual"] * df["RemodAge"]
    df["Qual_TotRms"] = df["OverallQual"] * df["TotRmsAbvGrd"]

    # 効率
    df["SF_per_room"] = df["GrLivArea"] / df["TotRmsAbvGrd"]
    df["Bath_per_room"] = df["TotalBath"] / df["TotRmsAbvGrd"]

    # 地下
    df["BsmtQual_TotalSF"] = df["TotalBsmtSF"] * df["OverallQual"]
    df["Bsmt_Liv"] = df["TotalBsmtSF"] * df["GrLivArea"]

    # ガレージ
    df["GarageFinish_num"] = df["GarageFinish"].map(GARAGE_FINISH_MAP).astype(float)
    df["GarageScore"] = df["GarageArea"] * df["GarageFinish_num"]
    df["Garage_SF"] = df["GarageArea"] * df["GrLivArea"]

    # フラグ
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    # 合成スコア
    df["ExteriorQual"] = (
        df["ExterQual"].map(QUAL_MAP).astype(float).fillna(0)
        + df["ExterCond"].map(QUAL_MAP).astype(float).fillna(0)
    )
    df["BsmtScore"] = (
        df["BsmtQual"].map(QUAL_MAP) + df["BsmtCond"].map(QUAL_MAP)
    )
    df["ConditionScore"] = (
        df["Condition1"].astype("category").cat.codes
        + df["Condition2"].astype("category").cat.codes
    )

    # ordinal encoding
    df["Functional_num"] = df["Functional"].map(FUNCTIONAL_MAP).astype(float)
    df["MSZoning_num"] = df["MSZoning"].astype("category").cat.codes
    df["SaleCondition_num"] = df["SaleCondition"].astype("category").cat.codes

    # 土地
    df["Lot_Score"] = df["LotArea"] * df["LotFrontage"].fillna(0)

    return df


def add_neighborhood_features(
    df: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Neighborhood 統計特徴量（train データの統計量を使って df に付与）"""
    df = df.copy()

    neigh_freq = train["Neighborhood"].value_counts()
    neigh_mean = train.groupby("Neighborhood")["GrLivArea"].mean()
    global_mean = train["GrLivArea"].mean()

    df["Neighborhood_freq"] = df["Neighborhood"].map(neigh_freq).fillna(0).astype(float)
    df["GrLivArea_Neigh_mean"] = df["Neighborhood"].map(neigh_mean).fillna(global_mean).astype(float)
    df["GrLivArea_ratio"] = df["GrLivArea"] / df["GrLivArea_Neigh_mean"]

    return df


def kfold_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    target: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """KFold Target Encoding（リーク防止）"""
    train_df = train_df.copy()
    test_df = test_df.copy()

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    train_encoded = pd.Series(index=train_df.index, dtype=float)
    global_mean = train_df[target].mean()

    test_target_mean = train_df.groupby(col)[target].mean()
    test_encoded = test_df[col].astype(str).map(test_target_mean).fillna(global_mean).astype(float)

    for _, (fit_idx, val_idx) in enumerate(kf.split(train_df)):
        fit_df = train_df.iloc[fit_idx]
        val_df = train_df.iloc[val_idx]
        fold_target_mean = fit_df.groupby(col)[target].mean()
        fold_encoded = val_df[col].astype(str).map(fold_target_mean).fillna(global_mean).astype(float)
        train_encoded.iloc[val_idx] = fold_encoded.values

    return train_encoded, test_encoded


def add_target_encoding_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "SalePrice",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Neighborhood の KFold Target Encoding と派生特徴量を追加"""
    train = train.copy()
    test = test.copy()

    train["Neighborhood_TE"], test["Neighborhood_TE"] = kfold_target_encode(
        train_df=train,
        test_df=test,
        col="Neighborhood",
        target=target_col,
    )

    for df in [train, test]:
        df["Neighborhood_Qual"] = df["Neighborhood_TE"] * df["OverallQual"]
        df["Neighborhood_SF"] = df["Neighborhood_TE"] * df["GrLivArea"]
        df["Neighborhood_TotalQual_SF"] = df["Neighborhood_TE"] * df["TotalQual_SF"]

    return train, test


FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalSF_with_porch",
    "TotalBath",
    "TotalQual_SF",
    "Neighborhood_TotalQual_SF",
    "HouseAge",
    "RemodAge",
    "BsmtQual_TotalSF",
    "Neighborhood_Qual",
    "Neighborhood_SF",
    "GrLivArea_ratio",
    "GarageCars",
    "GarageArea",
    "GarageFinish_num",
    "GarageScore",
    "KitchenScore",
    "TotalQual",
    "HasFireplace",
    "Fireplaces",
    "TotalBsmtSF",
    "YearBuilt",
    "FullBath",
    "TotRmsAbvGrd",
    "YearRemodAdd",
    "LotFrontage",
    "MasVnrArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "BsmtFinSF1",
    "OverallCond",
    "LotArea",
    "Condition1",
    "LotShape",
    "LandContour",
    "Functional_num",
    "MSZoning_num",
    "SaleCondition_num",
    "ExteriorQual",
    "BsmtScore",
    "ConditionScore",
    "Qual_Remod",
    "Garage_SF",
    "Bsmt_Liv",
    "Lot_Score",
]
