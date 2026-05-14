import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_solar_data(file_path: str, target_col: str = 'target'):
"""
    ヘッダー付きExcelを読み込み、特徴量(X)と正解データ(y)に分離する。
    """
    # 1行目をヘッダーとして読み込む
    df = pd.read_excel(file_path)
    
    # 特徴量として使用するカラムを明示的に指定
    feature_cols = [
        'Pvis[W/m^2]', 'Pnir[W/m^2]', 'Ain[deg]', 
        'Tmod[k]', 'Th[nm]', 'Bpsk[eV]', 'Terminals'
    ]
    
    X = df[feature_cols]
    y = df['Yield[W/m^2]']
    
    return X, y

def apply_scaling(X, scaler_path: str = 'checkpoints/scalers.pkl', is_train: bool = False):
    """StandardScalerの適用。学習時はsave、推論時はloadする"""
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    return X_scaled, scaler
