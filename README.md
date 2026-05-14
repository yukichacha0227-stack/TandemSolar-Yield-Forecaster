# TandemSolar-Yield-Forecaster
Deep learning-based yield prediction for Perovskite/Silicon tandem solar cells using actual meteorological data and theoretical device parameters. (全国の発電量適地予測に向けた、実測気象データと理論デバイス物性を統合した高精度予測モデル)

本プロジェクトは、ペロブスカイト/シリコンタンデム太陽電池の発電量（Yield）を高精度に予測するための機械学習パイプラインです。
Denoising AutoEncoder (DAE) による自己教師あり学習、TabNet による深層学習、そして 5-Fold GBDT 群を用いたスタッキング・アンサンブルを組み合わせた、石河研究室（青山学院大学）の研究成果をベースとした実装です。

## 🚀 Key Features
- **Hybrid Architecture**: DAE と TabNet を用いた高次元な特徴量抽出プロセス。
- **5-Fold Stacking Ensemble**: LightGBM, XGBoost, CatBoost の 5-Fold アンサンブルを Lasso (Meta-Regressor) で統合。
- **Robustness**: データのノイズに強い DAE 潜在変数と、未知の気象データに対して高い汎化性能を持つアンサンブル手法の融合。
- **Developer Friendly**: 複雑な資産管理を `src/config.py` で一元化し、ワンコマンドでの推論・評価が可能。

## 1. 依存ライブラリのインストール
pip install -r requirements.txt

##  🗺  リポジトリ構成
```text
.
├── README.md               # プロジェクトの概要・環境構築・実行手順
├── requirements.txt        # 依存ライブラリ一覧（pytorch-tabnet, lightgbm等）
├── main.py                 # 推論・評価をワンコマンドで実行するスクリプト
├── models/
│   ├── __init__.py
│   ├── dae.py              # DAE(Denoising AutoEncoder)のクラス定義
│   └── stacking_models.py  # GBDTやLassoを含むスタッキング層の定義
├── src/
│   ├── __init__.py
|   |── config.py           # Fold番号を自動で付与
│   ├── data_loader.py      # Excel読み込み、前処理、スケーリング
│   ├── feature_extractor.py # DAEとTabNetを用いた特徴量抽出ロジック
│   └── trainer.py          # 学習・バリデーション用ループ関数
├── checkpoints/            # 学習済み重み（.pth, .pkl）の格納先
│   ├── dae_best.pth
│   ├── tabnet_model.zip
│   └── scalers.pkl
├── data/
│   └── sample_input.xlsx   # 実行確認用のダミーデータ（公開用）
└── notebooks/
    └── experiment.ipynb    # 試行錯誤の過程を残したノートブック（公開用）
```

## 📦 Model Checkpoints
本モデルの動作に必要な学習済み資産は、[Releases](https://github.com/yukichacha0227-stack/TandemSolar-Yield-Forecaster/releases/tag/v1.0.0) からダウンロードし、以下のディレクトリ構造に従って配置してください。
※ `.gitignore` により、これらのバイナリファイルはGit管理から除外されています。

```text
checkpoints/
├── dae_best_model.pth            # DAEの重み・設定・学習履歴（辞書形式）
├── tabnet_standalone_best.zip        # TabNetのモデル本体
├── scalers (1).pkl             # 前処理（StandardScaler）の統計量
└── stacking/               # スタッキング層のベースモデル群
    ├── lgbm.pkl            # LightGBM (Base model 1)
    ├── xgb.pkl             # XGBoost (Base model 2)
    ├── cat.pkl             # CatBoost (Base model 3)
    └── stacking_meta_assets.pth      # Lasso (Meta-regressor)
```

## 📊 Data
- `data/sample_input.xlsx`: 動作確認用のサンプルデータです。
  - 実際の推論を行う際は、このファイルと同じカラム構成の Excel ファイルを用意してください。
  - 本プロジェクトでは、Atlas データの形式に基づいた前処理を `src/data_loader.py` で行っています。
  - ### 📊 Input Data Format (Excel)
推論に使用する Excel ファイル (`data/sample_input.xlsx`) は以下のカラム構成である必要があります。

| Column Name | Description | Unit |
| :--- | :--- | :--- |
| `Pvis[W/m^2]` | Visible light intensity | W/m² |
| `Pnir[W/m^2]` | Near-infrared light intensity | W/m² |
| `Ain[deg]` | Incident angle | degree |
| `Tmod[k]` | Module temperature | K |
| `Th[nm]` | Perovskite thickness | nm |
| `Bpsk[eV]` | Bandgap | eV |
| `Terminals` | Number of terminals | - |
| `Yield[W/m^2]` | Power generation (Target) | W/m² |
