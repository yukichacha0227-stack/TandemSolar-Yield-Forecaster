"""
Core Source Module for Tandem Solar Yield Prediction
src パッケージの主要コンポーネントをエクスポートします。
"""

from .config import get_stacking_assets
from .data_loader import load_solar_data, apply_scaling
from .feature_extractor import FeatureExtractor

# バージョン情報の定義（プロジェクト管理の慣習）
__version__ = "1.0.0"

# 外部に公開するリストを明示
__all__ = [
    "get_stacking_assets",
    "load_solar_data",
    "apply_scaling",
    "FeatureExtractor"
]
