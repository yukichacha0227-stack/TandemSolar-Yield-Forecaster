# TandemSolar-Yield-Forecaster
Deep learning-based yield prediction for Perovskite/Silicon tandem solar cells using actual meteorological data and theoretical device parameters. (全国の発電量適地予測に向けた、実測気象データと理論デバイス物性を統合した高精度予測モデル)

## 📦 Model Checkpoints
本モデルの動作に必要な学習済み資産は、[Releases]([https://github.com/yukichacha0227-stack/TandemSolar-Yield-Forecaster/releases/tag/v1.0.0]) からダウンロードし、以下のディレクトリ構造に従って配置してください。
※ `.gitignore` により、これらのバイナリファイルはGit管理から除外されています。

```text
checkpoints/
├── dae_best.pth            # DAEの重み・設定・学習履歴（辞書形式）
├── tabnet_model.zip        # TabNetのモデル本体
├── scalers.pkl             # 前処理（StandardScaler）の統計量
└── stacking/               # スタッキング層のベースモデル群
    ├── lgbm.pkl            # LightGBM (Base model 1)
    ├── xgb.pkl             # XGBoost (Base model 2)
    ├── cat.pkl             # CatBoost (Base model 3)
    └── meta_lasso.pkl      # Lasso (Meta-regressor)
