import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from scipy.stats import ks_2samp, epps_singleton_2samp, wasserstein_distance


# Univariate analysis
def qc_univariate(df_real: pd.DataFrame, df_syn: pd.DataFrame):
    """
    Compare df_syn (synthetic data) vs df_real (real data) in each variable
    Return a DataFrame with metrics: KS, Epps-Singleton, Wasserstein.
    """
    cols = ['UCS', 'RD', 'TC', 'W', 'E', 'OB']
    rows = []
    for c in cols:
        x = df_real[c].dropna().values
        y = df_syn[c].dropna().values

        ks_stat, ks_p = ks_2samp(x, y, alternative='two-sided', mode='auto')
        #es_stat, es_p = epps_singleton_2samp(x, y)
        #wdist = wasserstein_distance(x, y)
        #q25, q75 = np.percentile(np.r_[x, y], [25, 75])
        #iqr = max(q75 - q25, 1e-12)

        mean_r, mean_s = np.mean(x), np.mean(y)
        std_r,  std_s  = np.std(x, ddof=1), np.std(y, ddof=1)

        rows.append({
            "variable": c,
            "KS_stat": ks_stat, "KS_p": ks_p,
            #"Epps_stat": es_stat, "Epps_p": es_p,
            #"Wasserstein": wdist,
            #"EMD_norm_IQR": wdist / iqr,
            
            "min_df_real": np.min(x), "min_df_syn": np.min(y),
            "max_df_real": np.max(x), "max_df_syn": np.max(y),

            "mean_df_real": mean_r,  "mean_df_syn":  mean_s,
            "std_df_real":  std_r,   "std_df_syn":   std_s,

            "Δmean_%": 100.0 * abs(mean_s - mean_r) / (abs(mean_r) + 1e-12),
            "Δstd_%":  100.0 * abs(std_s  - std_r)  / (abs(std_r)  + 1e-12)
        })
    out = pd.DataFrame(rows).sort_values("variable")
    return out

# Correlation charts
def correlation_chart(df_real, df_syn, save_path=None):
    cols = ['UCS', 'RD', 'TC', 'W', 'E', 'OB']
    corr_r = df_real[cols].corr()
    corr_s = df_syn[cols].corr()
    corr_d = corr_r - corr_s

    # Dataframe
    corr_df = (
        corr_d['OB']
        .drop('OB')
        .rename('corr_diff_OB')
        .to_frame()
        .assign(corr_absdiff_OB=lambda d: d['corr_diff_OB'].abs())
        .reset_index()
        .rename(columns={'index':'variable'})
    )

    # chart
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    sns.heatmap(corr_r, annot=True, fmt=".2f", cmap='vlag', cbar_kws={'label': 'Correlation'}, ax=axes[0])
    axes[0].set_title('Correlation (Real)')
    sns.heatmap(corr_s, annot=True, fmt=".2f", cmap='vlag', cbar_kws={'label': 'Correlation'}, ax=axes[1])
    axes[1].set_title('Correlation (Synthetic)')
    sns.heatmap(corr_d, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'ΔCorr'}, ax=axes[2])
    axes[2].set_title('|Δcorr|')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        return fig, corr_df

# KDE charts   
def kernel_chart(df_real, df_syn, save_path=None):
    cols = df_real.columns.tolist()
    fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(3*len(cols), 3))
    for ax, col in zip(axes, cols):
        sns.kdeplot(df_syn[col], fill=False, ax=ax, label="Synthetic", color="#367BAC")
        sns.kdeplot(df_real[col], fill=False, ax=ax, label="True", color="#DBB972")
        ax.set_xlim(df_real[col].min(), df_real[col].max())
        ax.set_title(f"Densidad de {col}")
        ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig

def kernel_chart_total(df_real, df_syn, df_hyb, save_path=None):
    cols = df_real.columns.tolist()
    fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=(3*len(cols), 3))
    for ax, col in zip(axes, cols):
        sns.kdeplot(df_syn[col], fill=False, ax=ax, label="Synthetic", color="#367BAC")
        sns.kdeplot(df_hyb[col], fill=False, ax=ax, label="Hybrid", color = "#76C6BA")
        sns.kdeplot(df_real[col], fill=False, ax=ax, label="True", color="#DBB972")
        ax.set_xlim(df_real[col].min(), df_real[col].max())
        ax.set_title(f"Densidad de {col}")
        ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig

# Evaluate the sample with the previous fuctions
def evaluate_and_merge(df_real: pd.DataFrame,
                       df_syn:  pd.DataFrame,
                       save_dir: str | None = None):
    """
    Runs the fuctions to evaluate the samples (KDE, Correlations, and univariate metrics)
    Returns the metrics: merged_df and a score to compare
    If save_dir is not None, saves png or csv there.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig_corr, corr_df = correlation_chart(
        df_real, df_syn,
        save_path=(os.path.join(save_dir, "correlations.png") if save_dir else None)
    )
    fig_kde = kernel_chart(
        df_real, df_syn,
        save_path=(os.path.join(save_dir, "kde.png") if save_dir else None)
    )
    uni = qc_univariate(df_real, df_syn)

    merged = uni.merge(corr_df, on="variable", how="left")
    if save_dir:
        merged.to_csv(os.path.join(save_dir, "qc_merged.csv"), index=False)

    # Score
    mean_abs_corrdiff_OB = merged.loc[merged['variable'].isin(['UCS','RD','TC','W','E']),
                                   'corr_absdiff_OB'].mean()
    mean_dmean = (merged['Δmean_%'].abs().mean())/100.0
    mean_dstd  = (merged['Δstd_%'].abs().mean())/100.0
    w_corr, w_mean, w_std = 0.6, 0.2, 0.2
    score = w_corr*mean_abs_corrdiff_OB + w_mean*mean_dmean + w_std*mean_dstd

    return merged, score