import numpy as np
import pandas as pd


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    欠損値を補完する。

    補完方針は列の意味に応じて分ける。
    - 欠損自体に意味があるカテゴリ列は "None"
    - 面積・台数など不存在を表せる数値列は 0
    - 品質系の一部は最頻値
    - LotFrontage は Neighborhood ごとの中央値で補完し、残りは全体中央値で補完

    最後に残った欠損は、数値列は中央値、カテゴリ列は最頻値で補完する。
    """
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

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        # LotFrontage は地域差の影響を受けやすいため、
        # Neighborhood ごとの中央値で補完し、残りは全体中央値で補完する
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # 上記ルールで補完しきれなかった欠損を補完
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    object 型の列を category 型へ変換する。

    カテゴリ列として明示的に扱うことで、主に LightGBM での利用や
    後続の特徴量処理を整理しやすくするための前処理。
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df
