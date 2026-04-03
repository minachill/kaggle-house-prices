# Kaggle House Prices

## Overview
This project aims to predict house prices using the Kaggle House Prices dataset.  
<!-- このプロジェクトはKaggleのHouse Pricesデータセットを用いて住宅価格を予測することを目的としています -->

## Dataset
- Source: Kaggle House Prices  
  <!-- データセットの出典 -->
- Training samples: 1,460  
  <!-- 学習データ数 -->
- Features: 79 variables  
  <!-- 特徴量の数 -->
- Target: SalePrice  
  <!-- 目的変数 -->

## Approach
- Selected numerical features without missing values  
  <!-- 欠損値のない数値特徴量を選択 -->
- Used LightGBM for regression  
  <!-- 回帰モデルとしてLightGBMを使用 -->
- Applied 5-fold cross validation  
  <!-- 5分割のクロスバリデーションを実施 -->

## Model
- Model: LightGBM  
  <!-- 使用モデル -->
- Features: MSSubClass, LotArea  
  <!-- 使用した特徴量 -->

## CV Score
- RMSE (validation): ~38,600  
  <!-- 検証データでのRMSE（平均的な予測誤差） -->

## Notes
- Currently building baseline model  
  <!-- 現在はベースラインモデル構築段階 -->
- Next steps include:
  - Log transformation of target variable  
    <!-- 目的変数の対数変換 -->
  - Feature engineering  
    <!-- 特徴量エンジニアリング -->
  - Model improvement  
    <!-- モデルの改善 -->

## Tech Stack
- Python  
- pandas, numpy  
- scikit-learn  
- LightGBM  
<!-- 使用技術一覧 -->
