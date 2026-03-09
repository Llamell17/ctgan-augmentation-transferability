import os, json
from pathlib import Path

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import torch

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from scipy.stats import ks_2samp, epps_singleton_2samp, wasserstein_distance
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# Create new DataFrames
def concat_dataframe(df_c0_train, df_c1_train, df_c0_synthetic, df_c1_synthetic):
    df_real_train = pd.concat([df_c0_train, df_c1_train], ignore_index=True)
    df_synthetic_train = pd.concat([df_c0_synthetic, df_c1_synthetic], ignore_index=True)
    df_hybrid_train = pd.concat([df_real_train, df_synthetic_train], ignore_index=True)
    return df_real_train, df_synthetic_train, df_hybrid_train

# Overbreak predictions
def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}

# funnel shape for DNN
def make_funnel_archs(first_layer_vals, layers_vals, decay=0.5, min_neurons=16):
    """
    first_layer_vals: iterable with neurons from the 1st layer, given by the paper
    layers_vals: iterable with N layers, given by the paper
    decay: reduction factor per layer
    min_neurons: minimum neurons
    """
    archs = []
    for n1 in first_layer_vals:
        for L in layers_vals:
            sizes = [n1]
            for _ in range(L-1):
                sizes.append(max(int(sizes[-1] * decay), min_neurons))
            archs.append(tuple(sizes))
    archs = sorted(set(archs), key=lambda t: (len(t), -t[0]))
    return archs

def run_experiment(df_train, df_test, df_hybrid_train, seed):
    # Create the training and test data
    feature_cols = ["UCS","RD","TC","W","E"]
    target_col = "OB"

    X_train, y_train = df_train[feature_cols], df_train[target_col]
    X_hybrid, y_hybrid = df_hybrid_train[feature_cols], df_hybrid_train[target_col]
    X_test,  y_test = df_test[feature_cols],  df_test[target_col]
    results = {}

    # DA-XGB with hybrid data
    daxgb = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', eval_metric="rmse",
                     reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
                     random_state=seed, tree_method="hist", n_jobs=-1),
        param_grid={
            "n_estimators": list(range(100, 301, 20)),
            "max_depth": list(range(1, 11, 1)),
            "min_child_weight": list(range(1, 11, 1))
        },
        scoring="r2", cv=3, n_jobs=-1
    )
    daxgb.fit(X_hybrid, y_hybrid)
    #print("Best parameters of model DA-XGB:", daxgb.best_params_)
    pred_daxgb = daxgb.predict(X_test)
    results["DA-XGB"] = metrics(y_test, pred_daxgb)

    # SHAP
    best_da = daxgb.best_estimator_
    X_bg = X_hybrid.sample(min(200, len(X_hybrid)), random_state=seed)
    expl = shap.Explainer(best_da, X_bg)
    
    # I think it because of the number of points
    shap_values = expl(X_hybrid)   
    shap.plots.beeswarm(shap_values) #max_display=10)
    shap.plots.bar(shap_values) #max_display=10)
    # Otherwise
    #sv_test = expl(X_test)
    #shap.plots.beeswarm(sv_test)
    #shap.plots.bar(sv_test)

    # XGBoost with real data
    xgb = GridSearchCV(
        XGBRegressor(objective='reg:squarederror', eval_metric="rmse",
                     reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
                     random_state=seed, tree_method="hist", n_jobs=-1),
        param_grid={
            "n_estimators": list(range(100, 301, 20)),
            "max_depth": list(range(1, 11, 1)),
            "min_child_weight": list(range(1, 11, 1))
        },
        scoring="r2", cv=3, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    #print("Best parameters of model XGB:", xgb.best_params_)
    pred_xgb = xgb.predict(X_test)
    results["XGB"] = metrics(y_test, pred_xgb)

    # RF with real data
    rf = GridSearchCV(
        RandomForestRegressor(random_state=seed, n_jobs=-1),
        param_grid={
            "n_estimators": list(range(100, 301, 20)),
            "max_depth": list(range(1, 11, 1)),
            #"min_samples_leaf": [1, 2, 3], "max_features": ["sqrt", 0.8]
        },
        scoring="r2", cv=3, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    #print("Best parameters of model RF:", rf.best_params_)
    pred_rf = rf.predict(X_test)
    results["RF"] = metrics(y_test, pred_rf)

    # DNN with real data
    # Funnel (embudo) básico siguiendo el paper (2 capas típicas)
    first_layer_vals = list(range(64, 257, 32))     # paper
    layers_vals = [1, 2, 3]                         # paper
    hls_options = make_funnel_archs(first_layer_vals, layers_vals, decay=0.5, min_neurons=16)
    # scaling
    dnn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("dnn", MLPRegressor(random_state=seed))
    ])

    param_grid_dnn = {
    "dnn__hidden_layer_sizes": hls_options,
    "dnn__max_iter": list(range(10, 201, 20)),    # epochs
    "dnn__batch_size": [16, 32, 48, 64],          # batch
    "dnn__activation": ["relu"],                  # Rectified Linear Unit, converges faster
    # "dnn__learning_rate_init": [1e-3, 5e-4],
    # "dnn__alpha": [1e-5, 1e-4, 1e-3],
    }
    
    dnn = GridSearchCV(dnn_pipe, param_grid=param_grid_dnn, scoring="r2", cv=3, n_jobs=-1)
    dnn.fit(X_train, y_train)
    #print("Best parameters of model DNN:", dnn.best_params_)
    pred_dnn = dnn.predict(X_test)
    results["DNN"] = metrics(y_test, pred_dnn)

    # DataFrame with results
    kpis = ["R2", "RMSE", "MAE", "MAPE"]
    df_kpi = pd.DataFrame(
        {model: [vals[k] for k in kpis] for model, vals in results.items()},
        index=kpis
    ).round(3)

    best_params = {
    "DA-XGB": daxgb.best_params_,
    "XGB":    xgb.best_params_,
    "RF":     rf.best_params_,
    "DNN":    dnn.best_params_,
    }

    preds = {
    "DA-XGB": pred_daxgb,
    "XGBoost": pred_xgb,
    "RF": pred_rf,
    "DNN": pred_dnn,
    }
    
    return df_kpi, best_params, preds

# Crossvalidation
def cv5(df_train, best_params, pac, seed): #, augment_ratio=0.25):
    feature_cols = ["UCS","RD","TC","W","E"]
    target_col   = "OB"

    kf  = KFold(n_splits=5, shuffle=True, random_state=seed)
    rec = {m: [] for m in ["XGB","RF","DNN","DA-XGB"]}

    base_xgb = dict(
        objective="reg:squarederror", eval_metric="rmse",
        tree_method="hist", reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
        random_state=seed, n_jobs=-1
    )

    for tr_idx, va_idx in kf.split(df_train):
        data_tr, data_va = df_train.iloc[tr_idx].copy(), df_train.iloc[va_idx].copy()

        # Clustering
        km = KMeans(n_clusters=2, random_state=seed, n_init="auto")
        labels_tr = km.fit_predict(data_tr[["UCS","RD"]])
        data_tr["cluster"] = labels_tr

        # Aumentation data CTGAN
        target_new = {0: 20, 1: 40}  # tus valores fijos por clúster        
        synth_parts = []
        for c, n_new in target_new.items():
            df_c = data_tr.loc[data_tr["cluster"] == c, feature_cols + [target_col]].copy()
            syn_c = fit_and_sample_ctgan(df_c, n_new, pac, seed, chart=False)
            synth_parts.append(syn_c)
            df_syn = pd.concat(synth_parts, ignore_index=True)
            data_hy = pd.concat([data_tr.drop(columns=["cluster"]), df_syn], ignore_index=True)

        # Split
        Xtr, ytr = data_tr[feature_cols], data_tr[target_col]
        Xhy, yhy = data_hy[feature_cols], data_hy[target_col]
        Xva, yva = data_va[feature_cols], data_va[target_col]

        # DA-XGB
        da_params = {**base_xgb, **best_params["DA-XGB"]}
        da_model  = XGBRegressor(**da_params).fit(Xhy, yhy)
        rec["DA-XGB"].append(metrics(yva, da_model.predict(Xva)))

        # XGB
        xgb_params = {**base_xgb, **best_params["XGB"]}
        xgb_model  = XGBRegressor(**xgb_params).fit(Xtr, ytr)
        rec["XGB"].append(metrics(yva, xgb_model.predict(Xva)))

        # RF
        rf_model = RandomForestRegressor(random_state=seed, n_jobs=-1, **best_params["RF"]).fit(Xtr, ytr)
        rec["RF"].append(metrics(yva, rf_model.predict(Xva)))

        # DNN
        dnn_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("dnn", MLPRegressor(random_state=seed))
        ])
        dnn_pipe.set_params(**best_params["DNN"])  # acepta dnn__*
        dnn_pipe.fit(Xtr, ytr)
        rec["DNN"].append(metrics(yva, dnn_pipe.predict(Xva)))

    # Metrics
    rows = []
    for model, lst in rec.items():
        dfm = pd.DataFrame(lst)
        rows.append({
            "Model": model,
            "R2_mean": dfm["R2"].mean(),   "R2_std": dfm["R2"].std(ddof=1),
            "RMSE_mean": dfm["RMSE"].mean(),"RMSE_std": dfm["RMSE"].std(ddof=1),
            "MAE_mean": dfm["MAE"].mean(), "MAE_std": dfm["MAE"].std(ddof=1),
            "MAPE_mean": dfm["MAPE"].mean(),"MAPE_std": dfm["MAPE"].std(ddof=1),
        })
    return pd.DataFrame(rows).set_index("Model").round(3)

# Charts
def plot_overbreak_comparison(df_test, preds_dict, title_suffix=""):
    """
    y_test: test data, real values
    preds_dict: prediction of each model with the best parameters
    """
    y_test = df_test[['OB']]

    # Format
    model_order = ["DA-XGB", "XGBoost", "RF", "DNN"]
    markers = {"DA-XGB":"o", "XGBoost":"^", "RF":"v", "DNN":"D"}
    linestyles = {"DA-XGB":"--", "XGBoost":"--", "RF":"--", "DNN":"--"}
    colours = {"DA-XGB":"#F94341", "XGBoost":"#005ED2", "RF":"#009C62", "DNN":"#B87BD8"}

    x = np.arange(1, len(y_test)+1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left chart: Overbreak vs No data
    ax = axes[0]
    ax.plot(x, y_test, '-s', color='#525252', label='Monitoring', linewidth=1.5, markersize=5)
    for m in model_order:
        if m in preds_dict:
            ax.plot(x, preds_dict[m],
                    linestyles[m], marker=markers[m], color=colours[m], label=m, linewidth=1.2, markersize=4, alpha=0.9)
    ax.set_xlabel('Data No.')
    ax.set_ylabel('Overbreak')
    ax.set_title('(a) Overbreak vs. Data No.' + (f' — {title_suffix}' if title_suffix else ''))
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)

    # Right chart: prediction vs test data
    ax = axes[1]
    vmin = min(np.nanmin(y_test), *[np.nanmin(np.asarray(v).ravel()) for v in preds_dict.values()])
    vmax = max(np.nanmax(y_test), *[np.nanmax(np.asarray(v).ravel()) for v in preds_dict.values()])
    ax.plot([vmin, vmax], [vmin, vmax], 'k-', linewidth=1)
    for m in model_order:
        if m in preds_dict:
            ax.scatter(y_test, preds_dict[m], color=colours[m], marker=markers[m], label=m, alpha=0.9)
    ax.set_xlabel('Monitoring')
    ax.set_ylabel('Prediction')
    ax.set_title('(b) Prediction vs. Monitoring' + (f' — {title_suffix}' if title_suffix else ''))
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.show()
    return fig, axes