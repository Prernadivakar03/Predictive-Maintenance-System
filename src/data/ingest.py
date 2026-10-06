import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import load_config, get_path

logger = get_logger("data_ingestion")

def generate_synthetic_dataset(num_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a schema-compliant fallback dataset adhering to AI4I 2020 Predictive Maintenance specs.
    Used if remote URL downloads fail due to network constraints.
    """
    logger.warning("Generating synthetic AI4I 2020 dataset for fallback reproducibility...")
    np.random.seed(seed)
    
    product_types = np.random.choice(['L', 'M', 'H'], size=num_samples, p=[0.6, 0.3, 0.1])
    product_ids = [f"{t}{i:05d}" for i, t in enumerate(product_types, start=1)]
    
    air_temp = np.random.normal(loc=300.0, scale=2.0, size=num_samples)
    process_temp = air_temp + np.random.normal(loc=10.0, scale=1.0, size=num_samples)
    rotational_speed = np.random.normal(loc=1500.0, scale=150.0, size=num_samples)
    torque = np.random.normal(loc=40.0, scale=10.0, size=num_samples)
    tool_wear = np.random.randint(low=0, high=250, size=num_samples)
    
    # Failure modes simulation
    twf = (tool_wear > 200) & (np.random.rand(num_samples) < 0.1)
    hdf = ((process_temp - air_temp) < 8.6) & (rotational_speed < 1380)
    pwf = (torque * rotational_speed * (2 * np.pi / 60) > 9000) | (torque * rotational_speed * (2 * np.pi / 60) < 3500)
    osf = (tool_wear * torque > 11000)
    rnf = np.random.rand(num_samples) < 0.001
    
    machine_failure = (twf | hdf | pwf | osf | rnf).astype(int)
    
    df = pd.DataFrame({
        "UDI": np.arange(1, num_samples + 1),
        "Product ID": product_ids,
        "Type": product_types,
        "Air temperature [K]": np.round(air_temp, 1),
        "Process temperature [K]": np.round(process_temp, 1),
        "Rotational speed [rpm]": np.round(rotational_speed).astype(int),
        "Torque [Nm]": np.round(torque, 1),
        "Tool wear [min]": tool_wear,
        "Machine failure": machine_failure,
        "TWF": twf.astype(int),
        "HDF": hdf.astype(int),
        "PWF": pwf.astype(int),
        "OSF": osf.astype(int),
        "RNF": rnf.astype(int)
    })
    
    logger.info(f"Synthetic dataset created: {len(df)} rows, failure rate: {df['Machine failure'].mean():.2%}")
    return df

def download_dataset(target_path: Path = None, force: bool = False) -> Path:
    """
    Downloads or retrieves the raw AI4I 2020 Predictive Maintenance dataset.
    
    Args:
        target_path: Destination path for saving raw CSV.
        force: If True, re-downloads even if target file exists.
        
    Returns:
        Path to the saved raw CSV file.
    """
    config = load_config()
    raw_dir = get_path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    if target_path is None:
        target_path = raw_dir / config["data"]["dataset_filename"]
        
    if target_path.exists() and not force:
        logger.info(f"Raw dataset already exists at '{target_path}'. Skipping download (idempotent execution).")
        return target_path
        
    urls = [config["data"]["source_url"]] + config["data"].get("fallback_urls", [])
    download_success = False
    
    for url in urls:
        logger.info(f"Attempting download from: {url}")
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200 and len(response.content) > 1000:
                with open(target_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Successfully downloaded dataset to '{target_path}'. Size: {len(response.content)} bytes.")
                download_success = True
                break
            else:
                logger.warning(f"Download from {url} returned status code {response.status_code}.")
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {str(e)}")
            
    if not download_success:
        logger.warning("All remote dataset download attempts failed. Generating schema-compliant dataset locally.")
        df_synthetic = generate_synthetic_dataset()
        df_synthetic.to_csv(target_path, index=False)
        logger.info(f"Saved synthetic dataset to '{target_path}'.")
        
    return target_path

def load_raw_data(file_path: Path = None) -> pd.DataFrame:
    """
    Loads raw CSV dataset into a Pandas DataFrame.
    """
    if file_path is None:
        config = load_config()
        file_path = get_path(config["data"]["raw_dir"]) / config["data"]["dataset_filename"]
        
    if not file_path.exists():
        logger.info(f"Dataset not found at '{file_path}'. Triggering ingestion...")
        file_path = download_dataset(target_path=file_path)
        
    logger.info(f"Loading raw dataset from '{file_path}'...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns.")
    return df

if __name__ == "__main__":
    logger.info("Executing Data Ingestion Pipeline...")
    csv_path = download_dataset(force=False)
    data = load_raw_data(csv_path)
    print(f"Data Ingestion Complete! Dataset Shape: {data.shape}")
