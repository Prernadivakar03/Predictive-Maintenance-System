import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, get_path
from src.utils.logger import get_logger
from src.data.ingest import load_raw_data

logger = get_logger("eda_analysis")

def run_eda_and_generate_figures():
    """
    Performs Exploratory Data Analysis on raw AI4I 2020 dataset,
    saves figures to reports/figures/, and logs statistical summaries.
    """
    config = load_config()
    figures_dir = get_path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_raw_data()
    logger.info(f"Loaded raw dataset for EDA. Shape: {df.shape}")
    
    # Set styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})
    
    # 1. Target Distribution Plot
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="Machine failure", data=df, hue="Machine failure", palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Target Distribution: Machine Failure (0 = Normal, 1 = Failure)", fontsize=12, fontweight='bold')
    plt.xlabel("Machine Failure Status")
    plt.ylabel("Count")
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f"{100 * height / total:.2f}%"
        ax.annotate(f"{height:,}\n({percentage})", (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
                    
    target_fig_path = figures_dir / "target_distribution.png"
    plt.tight_layout()
    plt.savefig(target_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {target_fig_path}")

    # 2. Sensor Feature Distributions
    sensor_cols = [
        "Air temperature [K]", "Process temperature [K]", 
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    for i, col in enumerate(sensor_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#3498db")
        axes[i].set_title(f"Distribution of {col}", fontweight='bold')
        
    # Categorical Type Distribution in the 6th subplot
    sns.countplot(x="Type", data=df, ax=axes[5], palette="Blues_d", hue="Type", legend=False)
    axes[5].set_title("Distribution of Quality Type", fontweight='bold')
    
    sensor_fig_path = figures_dir / "sensor_distributions.png"
    plt.tight_layout()
    plt.savefig(sensor_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {sensor_fig_path}")

    # 3. Correlation Heatmap
    num_cols = sensor_cols + ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    corr = df[num_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix of Sensor Features & Failure Targets", fontsize=12, fontweight='bold')
    
    corr_fig_path = figures_dir / "correlation_matrix.png"
    plt.tight_layout()
    plt.savefig(corr_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {corr_fig_path}")

    # 4. Feature vs Machine Failure (Boxplots)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    for i, col in enumerate(sensor_cols):
        sns.boxplot(x="Machine failure", y=col, data=df, ax=axes[i], palette=["#2ecc71", "#e74c3c"], hue="Machine failure", legend=False)
        axes[i].set_title(f"{col} by Failure Status", fontweight='bold')
        
    # Subplot 6: Type vs Failure Rate
    failure_by_type = df.groupby("Type")["Machine failure"].mean().reset_index()
    sns.barplot(x="Type", y="Machine failure", data=failure_by_type, ax=axes[5], palette="Oranges_d", hue="Type", legend=False)
    axes[5].set_title("Failure Rate by Quality Type", fontweight='bold')
    axes[5].set_ylabel("Failure Rate")
    
    feature_vs_fig_path = figures_dir / "feature_vs_failure.png"
    plt.tight_layout()
    plt.savefig(feature_vs_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {feature_vs_fig_path}")

    # 5. Failure Mode Breakdown Plot
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    mode_counts = df[failure_modes].sum().reset_index()
    mode_counts.columns = ["Failure Mode", "Count"]
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Failure Mode", y="Count", data=mode_counts, palette="Spectral", hue="Failure Mode", legend=False)
    plt.title("Breakdown of Specific Failure Modes (TWF, HDF, PWF, OSF, RNF)", fontsize=12, fontweight='bold')
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height + 2),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    modes_fig_path = figures_dir / "failure_modes_breakdown.png"
    plt.tight_layout()
    plt.savefig(modes_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {modes_fig_path}")

    logger.info("All EDA plots generated successfully!")
    
    # Generate Jupyter Notebook `.ipynb` file programmatically
    generate_ipynb_notebook(df)

def generate_ipynb_notebook(df: pd.DataFrame):
    """
    Creates notebooks/eda.ipynb programmatically containing text analysis and figures.
    """
    notebook_dir = get_path("notebooks")
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / "eda.ipynb"
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploratory Data Analysis: Predictive Maintenance System\n",
                "\n",
                "This notebook analyzes the **AI4I 2020 Predictive Maintenance Dataset** to discover telemetry patterns, sensor correlations, and feature distributions related to equipment failures."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from src.data.ingest import load_raw_data\n",
                "\n",
                "df = load_raw_data()\n",
                "print(f'Dataset Shape: {df.shape}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Dataset Overview & Data Types"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('--- Dataset Info ---')\n",
                "df.info()\n",
                "print('\\n--- Summary Statistics ---')\n",
                "df.describe()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Target Variable Analysis: Machine Failure\n",
                "\n",
                "The target variable `Machine failure` is highly imbalanced with only **3.39% failures** (339 positive cases out of 10,000 samples).\n",
                "\n",
                "![Target Distribution](../reports/figures/target_distribution.png)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Sensor Distributions & Features\n",
                "\n",
                "Analyzing sensor telemetries: Air temp [K], Process temp [K], Rotational speed [rpm], Torque [Nm], and Tool wear [min].\n",
                "\n",
                "![Sensor Distributions](../reports/figures/sensor_distributions.png)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Correlation Analysis\n",
                "\n",
                "Notice strong positive correlation between `Air temperature [K]` and `Process temperature [K]` ($r \\approx 0.88$), and negative correlation between `Rotational speed [rpm]` and `Torque [Nm]` ($r \\approx -0.88$).\n",
                "\n",
                "![Correlation Matrix](../reports/figures/correlation_matrix.png)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Failure Mode Breakdown & Key Insights\n",
                "\n",
                "Five specific failure modes exist: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failures (RNF).\n",
                "\n",
                "![Failure Modes](../reports/figures/failure_modes_breakdown.png)\n",
                "\n",
                "![Features vs Failure](../reports/figures/feature_vs_failure.png)"
            ]
        }
    ]
    
    notebook_content = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.11"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
        
    logger.info(f"Generated EDA notebook at: {notebook_path}")

if __name__ == "__main__":
    run_eda_and_generate_figures()
