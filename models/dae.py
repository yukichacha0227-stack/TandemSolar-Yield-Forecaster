import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

class DAE(nn.Module):
    """
    Denoising AutoEncoder (DAE) for tabular data feature extraction.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32):
        super(DAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # 入力データが0-1スケーリングされている想定
        )

        # 属性の初期化（計算結果保持用）
        self.history = None
        self.feature_importances_ = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """推論時にボトルネック層の特徴量を抽出する"""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    def save_assets(self, path: str, best_loss: float):
        """
        重みだけでなく、学習履歴や損失値もセットで保存する。
        """
        save_data = {
            'state_dict': self.state_dict(),
            'feature_importances': self.feature_importances_,
            'history': self.history,
            'best_loss': best_loss,
            'model_config': {
                'input_dim': self.encoder[0].in_features,
                'hidden_dim': self.encoder[0].out_features,
                'latent_dim': self.encoder[3].out_features
            }
        }
        torch.save(save_data, path)
        print(f"Model assets saved to {path}")

    @classmethod
    def load_from_assets(cls, path: str, device: str = 'cpu'):
        """
        保存された辞書からモデルを完全に復元する。
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        
        # モデル構造の再現
        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.history = checkpoint.get('history')
        model.feature_importances_ = checkpoint.get('feature_importances')
        model.to(device)
        model.eval()
        return model, checkpoint.get('best_loss')
