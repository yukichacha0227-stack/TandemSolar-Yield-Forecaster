import torch
import numpy as np
from models.dae import DAE
from pytorch_tabnet.tab_model import TabNetRegressor

class FeatureExtractor:
    def __init__(self, dae_path, tabnet_path, device='cpu'):
        self.device = device
        # DAEのロード
        self.dae_model, _ = DAE.load_from_assets(dae_path, device=device)
        # TabNetのロード
        self.tabnet_model = TabNetRegressor()
        self.tabnet_model.load_model(tabnet_path)

    def get_features(self, X_scaled):
        # DAEから潜在変数抽出
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        dae_latents = self.dae_model.get_latent_features(X_tensor).cpu().numpy()
        # TabNetから予測値抽出
        tabnet_preds = self.tabnet_model.predict(X_scaled).reshape(-1, 1)
        # 結合してスタッキング用特徴量にする
        return np.hstack([dae_latents, tabnet_preds])
