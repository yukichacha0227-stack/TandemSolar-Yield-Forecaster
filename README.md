# TandemSolar-Yield-Forecaster
Deep learning-based yield prediction for Perovskite/Silicon tandem solar cells using actual meteorological data and theoretical device parameters. (全国の発電量適地予測に向けた、実測気象データと理論デバイス物性を統合した高精度予測モデル)

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
