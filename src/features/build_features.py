import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Ensure sys.path includes project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config, get_path
from src.utils.logger import get_logger
from src.data.ingest import load_raw_data

logger = get_logger("feature_engineering")

class PredictiveMaintenanceFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for predictive maintenance domain-specific feature engineering.
    Calculates temperature differences, mechanical power, tool strain, and interaction ratios.
    """
    def __init__(self):
        pass
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. Temperature Difference: Difference between Process and Air temperatures
        if "Process temperature [K]" in X_out.columns and "Air temperature [K]" in X_out.columns:
            X_out["temp_difference"] = X_out["Process temperature [K]"] - X_out["Air temperature [K]"]
            X_out["temp_ratio"] = X_out["Process temperature [K]"] / (X_out["Air temperature [K]"] + 1e-5)
            
        # 2. Mechanical Power (kW): Torque [Nm] * Rotational speed [rpm] * (2 * pi / 60) / 1000
        if "Torque [Nm]" in X_out.columns and "Rotational speed [rpm]" in X_out.columns:
            power_watts = X_out["Torque [Nm]"] * X_out["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
            X_out["power_kw"] = power_watts / 1000.0
            X_out["speed_torque_ratio"] = X_out["Rotational speed [rpm]"] / (X_out["Torque [Nm]"] + 1e-5)
            
        # 3. Tool Wear Strain Index: Cumulative tool wear multiplied by active torque
        if "Tool wear [min]" in X_out.columns and "Torque [Nm]" in X_out.columns:
            X_out["tool_wear_torque"] = X_out["Tool wear [min]"] * X_out["Torque [Nm]"]
            
        return X_out

def create_preprocessing_pipeline() -> Tuple[Pipeline, list, list]:
    """
    Creates a scikit-learn preprocessing pipeline combining domain feature engineering,
    scaling for numerical features, and One-Hot Encoding for categorical features.
    
    Returns:
        Tuple of (pipeline, feature_names_list, dropped_columns)
    """
    config = load_config()
    val_config = config.get("validation", {})
    
    num_cols = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "temp_difference", "temp_ratio", "power_kw",
        "speed_torque_ratio", "tool_wear_torque"
    ]
    
    cat_cols = ["Type"]
    
    # Numerical pipeline: StandardScaler
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: OneHotEncoder
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine via ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')
    
    # Full preprocessing pipeline including feature engineer
    full_pipeline = Pipeline([
        ('feature_engineering', PredictiveMaintenanceFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

def split_data(df: pd.DataFrame, target_col: str = "Machine failure", random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into Train (70%), Validation (15%), and Test (15%) sets
    using Stratified Splitting to preserve class imbalance ratios.
    """
    logger.info("Splitting dataset into Train (70%), Validation (15%), and Test (15%) with stratification...")
    
    # Stage 1: Split into 70% Train and 30% Temp (Val + Test)
    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df[target_col]
    )
    
    # Stage 2: Split 30% Temp into 15% Val and 15% Test
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=random_state, stratify=df_temp[target_col]
    )
    
    logger.info(f"Split completed: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    logger.info(f"Target Failure Rates -> Train: {df_train[target_col].mean():.2%}, Val: {df_val[target_col].mean():.2%}, Test: {df_test[target_col].mean():.2%}")
    
    return df_train, df_val, df_test

def get_processed_feature_names(fitted_pipeline) -> list:
    """Extracts column feature names from fitted ColumnTransformer."""
    ct = fitted_pipeline.named_steps['preprocessor']
    num_cols = ct.transformers_[0][2]
    cat_encoder = ct.transformers_[1][1].named_steps['onehot']
    cat_cols = cat_encoder.get_feature_names_out(ct.transformers_[1][2]).tolist()
    return list(num_cols) + cat_cols

def preprocess_and_save(raw_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Executes the end-to-end preprocessing workflow:
    1. Loads raw dataset if not provided.
    2. Performs stratified Train/Val/Test split BEFORE fitting preprocessor.
    3. Fits pipeline on Train set ONLY (prevents data leakage).
    4. Transforms Train, Val, and Test sets.
    5. Saves processed sets and preprocessor artifact to data/processed/.
    """
    if raw_df is None:
        raw_df = load_raw_data()
        
    config = load_config()
    target_col = config["validation"]["target_column"]
    processed_dir = get_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Stratified Split
    df_train, df_val, df_test = split_data(raw_df, target_col=target_col)
    
    # Separate Features and Target
    X_train_raw = df_train.drop(columns=[target_col])
    y_train = df_train[target_col].values
    
    X_val_raw = df_val.drop(columns=[target_col])
    y_val = df_val[target_col].values
    
    X_test_raw = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].values
    
    # 2. Fit Preprocessing Pipeline ON TRAIN DATA ONLY
    logger.info("Fitting Preprocessing Pipeline on Training Data ONLY...")
    pipeline = create_preprocessing_pipeline()
    X_train_proc = pipeline.fit_transform(X_train_raw)
    
    # Transform Val and Test using fitted pipeline
    logger.info("Transforming Validation and Test Data using fitted pipeline...")
    X_val_proc = pipeline.transform(X_val_raw)
    X_test_proc = pipeline.transform(X_test_raw)
    
    # Extract Feature Names
    feature_names = get_processed_feature_names(pipeline)
    logger.info(f"Engineered and processed {len(feature_names)} features: {feature_names}")
    
    # Create Processed DataFrames
    df_train_proc = pd.DataFrame(X_train_proc, columns=feature_names)
    df_train_proc[target_col] = y_train
    
    df_val_proc = pd.DataFrame(X_val_proc, columns=feature_names)
    df_val_proc[target_col] = y_val
    
    df_test_proc = pd.DataFrame(X_test_proc, columns=feature_names)
    df_test_proc[target_col] = y_test
    
    # 3. Save Processed Artifacts
    artifacts = {
        "train": df_train_proc,
        "val": df_val_proc,
        "test": df_test_proc,
        "preprocessor": pipeline,
        "feature_names": feature_names
    }
    
    # Save CSVs
    df_train_proc.to_csv(processed_dir / "train.csv", index=False)
    df_val_proc.to_csv(processed_dir / "val.csv", index=False)
    df_test_proc.to_csv(processed_dir / "test.csv", index=False)
    
    # Save Joblib Artifacts
    joblib.dump(df_train_proc, processed_dir / "train.joblib")
    joblib.dump(df_val_proc, processed_dir / "val.joblib")
    joblib.dump(df_test_proc, processed_dir / "test.joblib")
    joblib.dump(pipeline, processed_dir / "preprocessor.joblib")
    
    logger.info(f"Saved processed data & preprocessor.joblib to: {processed_dir}")
    return artifacts

def load_preprocessor(artifact_path: Path = None):
    """
    Loads saved fitted preprocessor pipeline for inference.
    """
    if artifact_path is None:
        config = load_config()
        artifact_path = get_path(config["data"]["processed_dir"]) / "preprocessor.joblib"
        
    if not artifact_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {artifact_path}. Run build_features.py first.")
        
    return joblib.load(artifact_path)

if __name__ == "__main__":
    logger.info("Executing Feature Engineering and Preprocessing Pipeline...")
    artifacts = preprocess_and_save()
    print("Preprocessing Completed Successfully!")
    print(f"Engineered Features ({len(artifacts['feature_names'])}): {artifacts['feature_names']}")
    print(f"Train Shape: {artifacts['train'].shape}, Val Shape: {artifacts['val'].shape}, Test Shape: {artifacts['test'].shape}")
