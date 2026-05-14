import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# 自作モジュールのインポート
from src.config import get_stacking_assets
from src.data_loader import load_solar_data, apply_scaling
from src.feature_extractor import FeatureExtractor
from models.stacking_models import StackingEnsemble

# ==========================================
# 0. グローバル設定（ここを書き換えるだけでOK）
# ==========================================
DATA_PATH = 'data/sample_input.xlsx'
CHECKPOINT_DIR = "checkpoints"

# Release時の実ファイル名定義
DAE_FILE = "dae_best_model.pth"
TABNET_FILE = "tabnet_standalone_best.zip"
SCALER_FILE = "scalers (1).pkl"

def main():
    # ---------------------------------------------------------
    # 1. 資産（Checkpoints）パスの確定
    # ---------------------------------------------------------
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 各モデル・スケーラーのフルパス構築
    DAE_PATH = os.path.join(CHECKPOINT_DIR, DAE_FILE)
    TABNET_PATH = os.path.join(CHECKPOINT_DIR, TABNET_FILE)
    SCALER_PATH = os.path.join(CHECKPOINT_DIR, SCALER_FILE)
    
    # stacking用資産（5-Fold分 + Lasso）を一括取得
    stacking_assets = get_stacking_assets()

    print(f"--- Tandem Solar Yield Prediction Pipeline (Device: {DEVICE}) ---")

    # ---------------------------------------------------------
    # 2. データのロードと前処理
    # ---------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} が見つかりません。")
        return

    print("[1/4] Loading and Scaling input data...")
    X, y_true = load_solar_data(DATA_PATH)
    # 保存済みの Scaler を適用（推論モードのため is_train=False）
    X_scaled, _ = apply_scaling(X, scaler_path=SCALER_PATH, is_train=False)

    # ---------------------------------------------------------
    # 3. 深層学習による特徴量抽出 (DAE & TabNet)
    # ---------------------------------------------------------
    print("[2/4] Extracting Deep Learning features (DAE & TabNet)...")
    extractor = FeatureExtractor(DAE_PATH, TABNET_PATH, device=DEVICE)
    X_features = extractor.get_features(X_scaled)

    # ---------------------------------------------------------
    # 4. スタッキング・アンサンブルによる最終推論
    # ---------------------------------------------------------
    print("[3/4] Loading Stacking Ensemble (5-Fold) and Predicting...")
    ensemble = StackingEnsemble()
    
    # config から受け取ったパスのリストを流し込む
    ensemble.load_models(
        lgbm_paths=stacking_assets["lgbm"],
        xgb_paths=stacking_assets["xgb"],
        cat_paths=stacking_assets["cat"],
        meta_path=stacking_assets["meta"]
    )
    
    y_pred = ensemble.predict(X_features)

    # ---------------------------------------------------------
    # 5. 予測結果の出力と評価
    # ---------------------------------------------------------
    print("[4/4] Processing Final Results...")
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*45)
    print(f"🚀 FINAL EVALUATION RESULTS")
    print("-" * 45)
    print(f"MAE:      {mae:.8f}")
    print(f"R2 Score: {r2:.8f}")
    print("="*45)
    
    # 予測値のプレビュー
    preview_df = pd.DataFrame({
        'True_Value': y_true.values.flatten() if hasattr(y_true, 'values') else y_true,
        'Predicted': y_pred.flatten()
    })
    print("\n--- Prediction Preview (First 5 rows) ---")
    print(preview_df.head())
    print("\nInference process completed successfully.")

if __name__ == "__main__":
    main()
