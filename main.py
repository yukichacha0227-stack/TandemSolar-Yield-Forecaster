import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_loader import load_solar_data, apply_scaling
from src.feature_extractor import FeatureExtractor
from models.stacking_models import StackingEnsemble

def main():
    # 1. 基本設定（パスとデバイス）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_PATH = 'data/sample_input.xlsx'
    CHECKPOINT_DIR = 'checkpoints'
    
    DAE_PATH = os.path.join(CHECKPOINT_DIR, 'dae_best_model.pth')
    TABNET_PATH = os.path.join(CHECKPOINT_DIR, 'tabnet_standalone_best.zip')
    SCALER_PATH = os.path.join(CHECKPOINT_DIR, 'scalers (1).pkl')
    STACKING_DIR = os.path.join(CHECKPOINT_DIR, 'stacking')

    print(f"--- Tandem Solar Yield Prediction Pipeline (Device: {DEVICE}) ---")

    # 2. データのロードと前処理
    print("[1/4] Loading and Scaling data...")
    X, y_true = load_solar_data(DATA_PATH)
    X_scaled, _ = apply_scaling(X, scaler_path=SCALER_PATH, is_train=False)

    # 3. DAE + TabNet による中間特徴量の抽出
    print("[2/4] Extracting Deep Learning features (DAE & TabNet)...")
    extractor = FeatureExtractor(DAE_PATH, TABNET_PATH, device=DEVICE)
    X_features = extractor.get_features(X_scaled)

    # 4. スタッキング・アンサンブルによる最終推論
    print("[3/4] Performing Stacking Inference...")
    stacking_model = StackingEnsemble()
    # 保存されている各ベースモデルとLassoをロード
    # ※実際の運用では StackingEnsemble クラスに load_models を追加するとスマートです
    # 今回は簡易的に推論処理を実行
    y_pred = stacking_model.predict(X_features)

    # 5. 評価指標の算出
    print("[4/4] Evaluating results...")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n================ FINAL RESULTS ================")
    print(f"MAE:  {mae:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"R2 Score: {r2:.8f}")
    print("===============================================")

if __name__ == "__main__":
    main()
