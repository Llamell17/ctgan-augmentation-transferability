# Evaluating the Transferability of CTGAN-Based Data Augmentation Across Data-Generating Regimes

This repository contains the code used to reproduce and extend the data augmentation pipeline proposed by Liu et al. for predicting **overbreak (OB)** in blast-based tunnel construction.

The study evaluates the **transferability of a two-stage augmentation strategy** based on **K-means clustering and CTGAN synthetic data generation** when applied to a dataset generated under a different structural regime using **Latin Hypercube Sampling (LHS)**.

The objective is to investigate how synthetic augmentation behaves when the structural assumptions of the original method (strong imbalance and temporal ordering) are modified.

---

## Methodology Overview

The implemented workflow (right) reproduces the augmentation pipeline proposed by Liu et al. (left) and evaluates its behavior under different synthetic augmentation regimes.

![Methodology pipeline](results/methodology_pipeline.png)

---

## Repository Structure

Project/
│
├── README.md  
├── requirements.txt  
├── .gitignore  
│
├── 0_Data/                # Input datasets (not included)  
│   └── README.md  
│
├── 1_Clustering/          # Dataset partitioning and K-means clustering  
│   └── 1_Clustering.ipynb  
│
├── 2_CGTAN/               # CTGAN tuning and synthetic data generation  
│   └── 02_tune_ctgan.ipynb  
│
├── 3_Models/              # Model training and evaluation  
│   └── 03_models_cv.ipynb  
│
├── src/                   # Utility functions used across notebooks  
│   ├── ctgan_eval.py  
│   ├── ctgan_utils.py  
│   ├── io_utils.py  
│   └── modeling.py  
│
└── results/               # Figures generated in the analysis  
    ├── methodology_pipeline.png
    ├── correlation_drift.png
    ├── performance.png
    ├── shap_importance.png
    └── tail_analysis.png

---

## Experimental Pipeline

The implemented workflow consists of the following steps:

1. Sequential train/test split (80–20)  
2. K-means clustering using geological variables (UCS and RD)  
3. Independent CTGAN training per cluster  
4. Synthetic data generation under two augmentation regimes:
   - Experiment 1 – High Synthetic Volume
   - Experiment 2 – Structural Mimicry
5. Sensitivity analysis by scaling synthetic sample volume  
6. Construction of hybrid datasets (real + synthetic)  
7. XGBoost model training  
8. Predictive evaluation using:
   - R²
   - RMSE
   - MAE
   - MAPE
9. Structural fidelity analysis:
   - univariate statistics
   - kernel density estimation
   - correlation drift
10. Feature importance analysis using SHAP  
11. Tail performance analysis using RMSE for extreme OB regions (q10 and q90)

---

## Data Availability

The datasets used in this study are not included in this repository due to confidentiality constraints.

To run the code, place the required input files inside the `0_Data/` directory.

Expected files:

0_Data/
├── data.xlsx  
└── paper.xlsx  

---

## Reproducibility

All experiments were executed using a fixed random seed:

42

The seed was applied consistently across:

- K-means clustering  
- CTGAN training  
- synthetic sample generation  
- XGBoost model training  

---

## Author

Llamell Ailen Martinez Gorbik
