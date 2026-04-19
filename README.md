# Kaggle House Prices - Advanced Regression Techniques

住宅価格予測コンペティションにおいて、特徴量設計・モデリング・アンサンブルを段階的に行い、ベースライン（LB 0.21813）から **LB 0.12022** まで改善した分析記録である。

本リポジトリでは、精度改善の過程だけでなく、**不採用とした施策とその判断根拠**も含めて記録している。

## プロジェクト概要

| 項目 | 内容 |
|---|---|
| コンペ | [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) |
| 課題 | 79 の説明変数から住宅販売価格を予測する回帰問題 |
| 評価指標 | RMSE（予測値と実測値の対数に対する RMSE） |
| 最終スコア | LB 0.12022 / CV 0.11686 |
| 最終手法 | 5 モデルの単純平均アンサンブル（LightGBM, XGBoost, CatBoost, Lasso, ElasticNet） |

## アプローチ

### 分析の進め方

実際の分析は EDA → 前処理 → 特徴量 → モデリングという一般的な順番ではなく、**まずベースラインを作成し精度改善を繰り返し、精度改善が鈍化した段階で EDA に立ち戻る**という流れで進めた。
ノートブックの番号は最終的な整理順であり、実際の作業順とは異なる。

この進め方の中で、以下のような判断を行った。

- 派生特徴量だけでは精度が頭打ちになり、**元カラムの再追加**で最大の改善を得た
- EDA で発見した log 変換・外れ値クリップは LightGBM では効果がなく、**アルゴリズムの特性を踏まえた施策選択**の重要性を学んだ

### 各ノートブックの概要

| No. | ノートブック | 内容 | 主な成果 |
|---|---|---|---|
| 00 | Baseline | 最小構成（2変数 + LightGBM）でベースラインを構築 | CV 0.21155 / LB 0.21813 |
| 01 | EDA | 目的変数の分布、相関、外れ値、多重共線性、歪度、欠損値、Neighborhood 別価格差を確認 | 後続の前処理・特徴量設計の仮説を整理 |
| 02 | Preprocessing | 欠損値補完（4 方針）と型変換 | 欠損の意味に応じた補完で情報を保持 |
| 03 | Feature Engineering | 基本派生 → Neighborhood TE → 元カラム再追加 → 不採用施策の検証 | CV 0.21155 → 0.12913 / LB 0.21813 → 0.13057 |
| 04 | Modeling | 木系 3 モデル + 線形 3 モデルを比較、Optuna によるハイパーパラメータ最適化 | 各モデルの特性と予測傾向の違いを把握 |
| 05 | Ensemble | 予測相関の確認、単純平均 vs 加重平均の比較 | CV 0.11686 / LB 0.12022（単純平均を採用） |

### 特徴量エンジニアリングの方針

特徴量設計では、段階的にアプローチを変えながら検証を進め、各段階での CV スコアの推移と判断根拠を記録した。

| 段階 | 施策 | 前CV | 後CV |
|---|---|---:|---:|
| 1 | 基本派生特徴量（`TotalSF`, `Qual_TotalSF`, `HouseAge` 等） | 0.21155 | 0.14891 |
| 2 | Neighborhood TE + 交互作用特徴量 | 0.14891 | 0.14325 |
| 3 | 元カラムの再追加（`OverallCond`, `LotArea`, `YearBuilt` 等） | 0.14325 | 0.12913 |
| 4 | EDA を踏まえた追加検証（log 変換、外れ値クリップ、3way 交互作用） | 0.12913 | 改善なし → 不採用 |

段階 3 の元カラム再追加が最大の改善をもたらした。
派生特徴量で情報を要約することに注力しすぎ、元カラムが持つ生の情報を捨てていたことに気づいた転換点だった。

段階 4 では、EDA で発見した課題（歪度、外れ値、多重共線性）を施策に落としたが、いずれも LightGBM では改善につながらなかった。
木構造は単調変換に不変であり、分布補正の恩恵を受けにくいことを実験で確認した。

### アンサンブル

木系モデル（LightGBM, XGBoost, CatBoost）と線形モデル（Lasso, ElasticNet）の 5 モデルで単純平均アンサンブルを構成した。

| モデル | 系統 | 単体 LB |
|---|---|---:|
| ElasticNet | 線形 | 0.12555 |
| Lasso | 線形 | 0.12560 |
| CatBoost | 木系 | 0.12709 |
| XGBoost | 木系 | 0.12825 |
| LightGBM | 木系 | 0.12993 |
| **アンサンブル（単純平均）** | — | **0.12022** |

なお、Ridge も 04 で比較したが、線形モデル内で CV が最も低かったためアンサンブル候補から除外した。
木系と線形のtest 予測の相関が 0.975〜0.983 と完全には一致せず、この差がアンサンブルの改善幅につながった。
加重平均との差は 0.00001 にとどまったため、汎用性を重視して単純平均を採用した。

## スコア推移

ベースラインから最終スコアまでの主要な改善ポイントを整理する。

| 段階 | 施策 | CV | LB |
|---|---|---:|---:|
| Baseline | 2 変数 + LightGBM | 0.21155 | 0.21813 |
| 特徴量 2.1 | 基本派生特徴量 | 0.14891 | 0.15942 |
| 特徴量 2.2 | Neighborhood TE + 交互作用特徴量 | 0.14325 | 0.15169 |
| 特徴量 2.3 | 元カラム再追加 | 0.12913 | 0.13057 |
| Optuna | LightGBM ハイパーパラメータ最適化 | 0.12619 | 0.12993 |
| マルチモデル | 5 モデル比較 + Optuna 最適化 | 0.118〜0.126 | 0.125〜0.130 |
| Ensemble | 5 モデル単純平均 | 0.11686 | **0.12022** |

## プロジェクト構成

```
kaggle-house-prices/
├── data/                  # 学習・テストデータ
├── notebooks/
│   ├── 00_baseline.ipynb           # ベースライン作成
│   ├── 01_EDA.ipynb                # 探索的データ分析
│   ├── 02_preprocessing.ipynb      # 前処理（欠損値補完・型変換）
│   ├── 03_feature_engineering.ipynb # 特徴量エンジニアリング
│   ├── 04_modeling.ipynb           # モデリング・Optuna チューニング
│   └── 05_ensemble.ipynb           # アンサンブル
├── src/
│   ├── utils.py            # CV・OOF・提出ファイル生成等の共通関数
│   ├── features.py         # 特徴量生成・FEATURES リスト
│   └── preprocessing.py    # 欠損値補完・型変換
├── submissions/            # Kaggle 提出ファイル
├── requirements.txt        # ライブラリ一覧
└── README.md
```

**ノートブック** は分析の過程と判断根拠を記録する場所、**`src/`** は最終的な実装を集約する場所として役割を分けている。

## 使用技術

| カテゴリ | 技術 |
|---|---|
| 言語 | Python 3.13 |
| 木系モデル | LightGBM, XGBoost, CatBoost |
| 線形モデル | Lasso, ElasticNet, Ridge |
| ハイパーパラメータ最適化 | Optuna |
| 前処理・評価 | scikit-learn, NumPy, pandas, SciPy |
| 可視化 | Matplotlib, Seaborn |
| 開発環境 | Jupyter Notebook |
| バージョン管理 | Git / GitHub |

## 実行方法

1. リポジトリをクローンする

```bash
git clone https://github.com/minachill/kaggle-house-prices.git
cd kaggle-house-prices
```

2. データ・提出ファイル用のフォルダを作成する

```bash
mkdir -p data submissions
```

3. [Kaggle のデータページ](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)から `train.csv`, `test.csv` をダウンロードし、`data/` に配置する

4. 必要ライブラリをインストールする

```bash
pip install -r requirements.txt
```

5. ノートブックを番号順（`00_baseline.ipynb` → `05_ensemble.ipynb`）に実行する
提出ファイルは `submissions/` に生成される

※ `00_baseline.ipynb` は `src/` に依存せず、ノートブック単体で実行できる。
`01` 以降のノートブックでは `src/` の関数を利用している。