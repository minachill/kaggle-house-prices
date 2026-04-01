import numpy as np
import pandas as pd


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値補完"""
    df = df.copy()

    none_cols = [
        "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "MasVnrType",
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath",
        "MasVnrArea",
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    mode_fill_cols = ["ExterQual", "KitchenQual"]
    for col in mode_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # LotFrontage: Neighborhood 別中央値 → 全体中央値
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # 残欠損
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """object 列を LightGBM 用 category 型に変換"""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df
