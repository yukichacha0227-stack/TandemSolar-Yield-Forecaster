import os

# 基本ディレクトリの設定
CHECKPOINT_DIR = "checkpoints"
STACKING_DIR = os.path.join(CHECKPOINT_DIR, "stacking")
NUM_FOLDS = 5

# --- Stacking資産の定義 ---
# Releaseでの実際のファイル名に基づいたベース名称
STACKING_CONFIG = {
    "META": "stacking_meta_assets.pth",
    "LGBM_BASE": "lgbm_fold",
    "XGB_BASE": "xgb_fold",
    "CAT_BASE": "cat_fold"
}

def get_stacking_assets():
    """
    5-Fold分の各モデルパスと、Lassoのパスを整理して返す。
    main.pyはこの関数を呼ぶだけで、必要な全パスを取得できます。
    """
    # メタモデル（Lasso）のフルパス
    meta_path = os.path.join(STACKING_DIR, STACKING_CONFIG["META"])
    
    # 5-Fold分のパスリストを生成
    lgbm_paths = [os.path.join(STACKING_DIR, f"{STACKING_CONFIG['LGBM_BASE']}{i}.pkl") for i in range(NUM_FOLDS)]
    xgb_paths  = [os.path.join(STACKING_DIR, f"{STACKING_CONFIG['XGB_BASE']}{i}.pkl") for i in range(NUM_FOLDS)]
    cat_paths  = [os.path.join(STACKING_DIR, f"{STACKING_CONFIG['CAT_BASE']}{i}.pkl") for i in range(NUM_FOLDS)]
    
    return {
        "lgbm": lgbm_paths,
        "xgb": xgb_paths,
        "cat": cat_paths,
        "meta": meta_path
    }
