import numpy as np
import joblib
import torch
import os
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso

class StackingEnsemble:
    """
    Stacking Ensemble Model: 
    5-FoldのGBDTベースモデル群と、Lasso（Meta-Regressor）を統合したクラス。
    """
    def __init__(self):
        # 5-Fold分の各アルゴリズムを格納するリスト
        self.lgbm_models = []
        self.xgb_models = []
        self.cat_models = []
        
        # メタモデル（Lasso）
        self.meta_model = None

    def load_models(self, lgbm_paths, xgb_paths, cat_paths, meta_path):
        """
        src/config.py から生成されたパスリストを用いて、全ての学習済み資産をロードする。
        
        Args:
            lgbm_paths (list): LightGBM (fold0-4) のパスリスト
            xgb_paths (list): XGBoost (fold0-4) のパスリスト
            cat_paths (list): CatBoost (fold0-4) のパスリスト
            meta_path (str): stacking_meta_assets.pth へのパス
        """
        # ベースモデル（.pkl）を一括ロード
        self.lgbm_models = [joblib.load(p) for p in lgbm_paths]
        self.xgb_models = [joblib.load(p) for p in xgb_paths]
        self.cat_models = [joblib.load(p) for p in cat_paths]
        
        # メタモデル（.pth）のロード
        # 辞書形式 {'model': lasso_obj, ...} で保存されている場合を想定
        checkpoint = torch.load(meta_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.meta_model = checkpoint['model']
        else:
            self.meta_model = checkpoint
            
        print(f"Successfully loaded 5-fold ensemble models and meta-assets.")

    def predict(self, X):
        """
        5-Foldそれぞれの予測結果を平均し、それをメタモデルに入力して最終結果を得る。
        """
        if not self.meta_model:
            raise ValueError("Models are not loaded. Call load_models() first.")

        # 各アルゴリズムで、5つのFoldの予測値を平均 (アンサンブル)
        # axis=0 で平均をとることで、データ数×1の予測ベクトルを得る
        p1 = np.mean([model.predict(X) for model in self.lgbm_models], axis=0)
        p2 = np.mean([model.predict(X) for model in self.xgb_models], axis=0)
        p3 = np.mean([model.predict(X) for model in self.cat_models], axis=0)
        
        # 3つの予測値を横に結合（Lassoへの入力特徴量作成）
        meta_features = np.column_stack([p1, p2, p3])
        
        # メタモデルによる最終的な発電量予測
        return self.meta_model.predict(meta_features)
