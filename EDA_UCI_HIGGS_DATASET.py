
"""
Exploratory Data Analysis (EDA) script for the
 Higgs Boson dataset (UCI Machine Learning Repository).


"""

import argparse
import os
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from higgs_loader import load_data

# Use a clean visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

# Folder where all plots will be saved
OUTPUT_DIR = "results/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(fig, path):
    """
    A helper function that safely saves a plot.
    If a plot fails to save normally, it tries a fallback.
    """
    try:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        try:
            plt.savefig(path, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Could not save plot: {path} — {e}")
            plt.close("all")


def perform_eda(nrows=500000):
    """
    This function performs the full EDA process:
    - Loads the data
    - Shows summary statistics
    - Creates many useful plots
    - Saves everything in results/eda/
    """

    print(f"\nLoading up to {nrows:,} rows...")

    # Try loading the dataset
    try:
        df = load_data(nrows=nrows)
    except Exception as e:
        print("Could not load dataset:", e)
        traceback.print_exc()
        return

    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")

    # -------------------------------------------------------------
    # Basic information about the dataset
    # -------------------------------------------------------------
    info_file = os.path.join(OUTPUT_DIR, "0_info.txt")
    with open(info_file, "w") as f:
        df.info(buf=f)
    print("Saved dataset info.")

    df.describe().transpose().to_csv(os.path.join(OUTPUT_DIR, "1_describe.csv"))
    print("Saved summary statistics.")

    df.isnull().sum().to_csv(os.path.join(OUTPUT_DIR, "2_missing_values.csv"))
    print("Saved missing value report.")

    # Make sure the target column exists
    if "label" not in df.columns:
        print("The dataset does not contain a 'label' column. Stopping EDA.")
        return

    # -------------------------------------------------------------
    # Choose some features to visualize
    # -------------------------------------------------------------
    default_features = [
        "lepton_pT", "lepton_eta", "missing_energy_magnitude",
        "jet1_pt", "jet1_b-tag", "m_jj", "m_wwbb"
    ]
    features = [f for f in default_features if f in df.columns]

    if not features:
        # Use first numeric columns if defaults are missing
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col != "label"][:7]

    print("Using these features:", features)

    # -------------------------------------------------------------
    # 1. Plot target distribution
    # -------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(x="label", data=df, ax=ax)
        ax.set_title("Target Label Distribution")
        save_plot(fig, os.path.join(OUTPUT_DIR, "1_target_distribution.png"))
    except Exception as e:
        print("Error while plotting target distribution:", e)

    # -------------------------------------------------------------
    # 2. Feature histograms
    # -------------------------------------------------------------
    try:
        num_feats = len(features)
        cols = 3
        rows = (num_feats + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            sns.histplot(df, x=feature, hue="label", element="step",
                         stat="density", common_norm=False, ax=axes[i])
            axes[i].set_title(feature)

        for ax in axes[num_feats:]:
            ax.axis("off")

        plt.tight_layout()
        save_plot(fig, os.path.join(OUTPUT_DIR, "2_feature_histograms.png"))
    except Exception as e:
        print("Error while creating histograms:", e)

    # -------------------------------------------------------------
    # 3. Correlation matrix
    # -------------------------------------------------------------
    try:
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        save_plot(fig, os.path.join(OUTPUT_DIR, "3_correlation_matrix.png"))
        corr.to_csv(os.path.join(OUTPUT_DIR, "3_correlation_matrix.csv"))
    except Exception as e:
        print("Error computing correlation matrix:", e)

    # -------------------------------------------------------------
    # 4. Missing value heatmap
    # -------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, ax=ax)
        ax.set_title("Missing Value Heatmap")
        save_plot(fig, os.path.join(OUTPUT_DIR, "4_missing_value_heatmap.png"))
    except Exception as e:
        print("Error creating missing-value heatmap:", e)

    # -------------------------------------------------------------
    # 5. Correlation with target
    # -------------------------------------------------------------
    try:
        if "label" in corr.columns:
            label_corr = corr["label"].drop("label", errors="ignore").sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(label_corr))))
            label_corr.plot(kind="barh", ax=ax)
            ax.set_title("Correlation with Target Label")
            save_plot(fig, os.path.join(OUTPUT_DIR, "5_corr_with_target.png"))
            label_corr.to_csv(os.path.join(OUTPUT_DIR, "5_corr_with_target.csv"))
    except Exception as e:
        print("Error while computing correlation with target:", e)

    # -------------------------------------------------------------
    # 6. KDE plots (sampled)
    # -------------------------------------------------------------
    try:
        sample_n = min(20000, max(2000, len(df) // 50))
        sample_df = df.sample(sample_n, random_state=42)

        n = min(len(features), 6)
        rows = (n + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 4 * rows))
        axes = axes.flatten()

        for i, feat in enumerate(features[:n]):
            sns.kdeplot(sample_df, x=feat, hue="label", fill=True, ax=axes[i])
            axes[i].set_title(f"KDE: {feat}")

        for ax in axes[n:]:
            ax.axis("off")

        save_plot(fig, os.path.join(OUTPUT_DIR, "6_kde_plots.png"))
    except Exception as e:
        print("Error creating KDE plots:", e)

    # -------------------------------------------------------------
    # 7. Boxplots for outlier comparison
    # -------------------------------------------------------------
    try:
        num_feats = len(features)
        cols = 3
        rows = (num_feats + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = axes.flatten()

        for i, feat in enumerate(features):
            sns.boxplot(x="label", y=feat, data=df, ax=axes[i])
            axes[i].set_title(f"Boxplot: {feat}")

        for ax in axes[num_feats:]:
            ax.axis("off")

        save_plot(fig, os.path.join(OUTPUT_DIR, "7_boxplots_by_label.png"))
    except Exception as e:
        print("Error creating boxplots:", e)

    # -------------------------------------------------------------
    # 8. Pairplot (sampled)
    # -------------------------------------------------------------
    try:
        pair_size = min(5000, sample_n)
        pair_df = df.sample(pair_size, random_state=42)
        pair_features = ["label"] + features[:4]

        pp = sns.pairplot(pair_df[pair_features], hue="label", corner=True)
        pp.fig.suptitle("Pairplot (Sample)", y=1.02)
        pp.fig.savefig(os.path.join(OUTPUT_DIR, "8_pairplot_sample.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("Pairplot skipped:", e)

    # -------------------------------------------------------------
    # 9. Feature variance 
    # -------------------------------------------------------------
    try:
        variance = df.var().sort_values(ascending=False)
        top_var = variance.head(15)

        fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(top_var))))
        top_var.plot(kind="barh", ax=ax)
        ax.set_title("Top 15 Most Variable Features")
        save_plot(fig, os.path.join(OUTPUT_DIR, "9_feature_variance.png"))
        top_var.to_csv(os.path.join(OUTPUT_DIR, "9_feature_variance.csv"))
    except Exception as e:
        print("Error computing feature variance:", e)

    # -------------------------------------------------------------
    # 10. Simple scatter plot for two important features
    # -------------------------------------------------------------
    try:
        if "lepton_pT" in df.columns and "missing_energy_magnitude" in df.columns:
            scatter_df = df.sample(min(15000, sample_n), random_state=42)

            fig, ax = plt.subplots(figsize=(8, 8))
            sns.scatterplot(
                data=scatter_df,
                x="lepton_pT",
                y="missing_energy_magnitude",
                hue="label",
                s=12,
                alpha=0.6,
                ax=ax
            )
            ax.set_title("lepton_pT vs missing_energy_magnitude")
            save_plot(fig, os.path.join(OUTPUT_DIR, "10_jointplot_scatter.png"))
    except Exception as e:
        print("Error creating scatter plot:", e)

    # -------------------------------------------------------------
    # 11. Basic outlier detection using boxplots
    # -------------------------------------------------------------
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != "label"]
        check_cols = num_cols[:6]  # just a few for clarity

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, col in enumerate(check_cols):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f"Outliers: {col}")

        for ax in axes[len(check_cols):]:
            ax.axis("off")

        save_plot(fig, os.path.join(OUTPUT_DIR, "11_outlier_iqr.png"))
    except Exception as e:
        print("Error computing outliers:", e)

    # -------------------------------------------------------------
    # Save a quick summary file
    # -------------------------------------------------------------
    try:
        summary = pd.DataFrame({
            "rows": [len(df)],
            "columns": [df.shape[1]],
            "positive_labels": [int(df["label"].sum())],
            "negative_labels": [int((df["label"] == 0).sum())],
        })
        summary.to_csv(os.path.join(OUTPUT_DIR, "12_summary_overview.csv"), index=False)
    except Exception as e:
        print("Error saving summary file:", e)

    print("\nEDA complete. All results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" EDA for Higgs dataset")
    parser.add_argument("--nrows", type=int, default=500000,
                        help="Number of rows to load (default: 500k)")
    args = parser.parse_args()

    perform_eda(nrows=args.nrows)
