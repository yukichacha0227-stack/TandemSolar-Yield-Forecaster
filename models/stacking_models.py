import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
import joblib
import os

class StackingEnsemble:
    """
    Stacking Ensemble Model: 
    Combines LightGBM, XGBoost, and CatBoost using Lasso as a Meta-Regressor.
    """
    def __init__(self, lasso_alpha=0.1):
        # Base Models
        self.lgbm = LGBMRegressor(random_state=42)
        self.xgb = XGBRegressor(random_state=42)
        self.cat = CatBoostRegressor(random_state=42, verbose=0)
        
        # Meta Model
        self.meta_model = Lasso(alpha=lasso_alpha)
        
    def fit(self, X, y):
        """全てのベースモデルとメタモデルを訓練する"""
        self.lgbm.fit(X, y)
        self.xgb.fit(X, y)
        self.cat.fit(X, y)
        
        # 各モデルの予測値を特徴量としてスタック
        p1 = self.lgbm.predict(X)
        p2 = self.xgb.predict(X)
        p3 = self.cat.predict(X)
        
        meta_features = np.column_stack([p1, p2, p3])
        self.meta_model.fit(meta_features, y)
        print("Stacking Ensemble training completed.")

    def predict(self, X):
        """スタッキングによる最終的な発電量予測を行う"""
        p1 = self.lgbm.predict(X)
        p2 = self.xgb.predict(X)
        p3 = self.cat.predict(X)
        
        meta_features = np.column_stack([p1, p2, p3])
        return self.meta_model.predict(meta_features)

    def save_models(self, folder_path):
        """各モデルを個別に保存（保守性を考慮）"""
        os.makedirs(folder_path, exist_ok=True)
        joblib.dump(self.lgbm, os.path.join(folder_path, 'lgbm.pkl'))
        joblib.dump(self.xgb, os.path.join(folder_path, 'xgb.pkl'))
        joblib.dump(self.cat, os.path.join(folder_path, 'cat.pkl'))
        joblib.dump(self.meta_model, os.path.join(folder_path, 'meta_lasso.pkl'))
        print(f"All models saved to {folder_path}")
