# 🏆 alrIEEEna'26 — ML Challenge
## IEEE SB, Graphic Era Hill University
### Binary Fault Detection: Normal (0) vs Faulty (1)

---

##  Problem Statement
Classify device operational status using 47 sensor 
readings (F01-F47) as either Normal (0) or Faulty (1).

Dataset: 43,776 training samples | 10,944 test samples

---

##  Repository Structure
```
MLareenacompetition/
├── FINAL.csv                  # Final predictions (10,944 rows)
├── MLareenacompetition.ipynb  # Complete ML pipeline notebook
└── README.md                  # Project description
```

---

##  Data Analysis Findings

| Issue | Finding | Fix Applied |
|-------|---------|-------------|
| Duplicate Rows | 738 duplicates found | Removed → 43,038 clean rows |
| Outliers | All 47 features affected | RobustScaler applied |
| Negative Values | 10 features had negatives | Tracked via row_neg_count |
| Class Imbalance | 1.49:1 ratio | class_weight=balanced |

---

## ⚙️ Feature Engineering (47 → 62 Features)

Created 15 new statistical features per row:

| Feature | Purpose |
|---------|---------|
| row_max | Captures F31/F32 fault spikes (up to 1000!) |
| row_neg_count | Tracks negative sensor readings |
| row_energy | Amplifies strong signals (sum of squares) |
| row_range | Large for Faulty, small for Normal |
| row_std | Faulty = unstable, Normal = stable |
| row_skew | Asymmetry in sensor readings |
| row_kurt | Extreme peak detection |
| row_iqr | Robust spread measure |
| row_mean | Overall signal level |
| row_median | Outlier-robust average |
| row_l1norm | Total magnitude |
| row_q25/q75 | Quartile based features |
| row_nearzero | Normal devices have more zero readings |

---

##  Model Training & Results

### Individual Models (Validation Set)
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| XGBoost | 0.9843 | 0.9843 |
| LightGBM | 0.9812 | 0.9812 |
| RandomForest | 0.9803 | 0.9802 |
| ExtraTrees | 0.9790 | 0.9789 |
| CatBoost | 0.9758 | 0.9758 |

### Hyperparameter Tuning (Optuna)
| Model | CV F1 Score |
|-------|------------|
| XGBoost tuned | 0.9837 |
| LightGBM tuned | 0.9861 |

### Final Ensemble (Soft Voting)
| Metric | Score |
|--------|-------|
| **Accuracy** | **0.9884** |
| **F1 Score** | **0.9884** |
| **AUC Score** | **0.9993** |

---

## 📊 Key Visualizations

1. Class Distribution (Pie + Bar)
2. Feature Distributions by Class (F01-F12)
3. Outlier Box Plots (Top 10 features)
4. Negative Values Analysis
5. Feature Correlation Heatmap
6. Mutual Information Feature Ranking
7. PCA 2D Projection

---

## 🔑 Key Insights

- **F01 = Most Important Feature** (MI Score: 0.14)
- **F41-F42 = Perfect -1.0 Correlation** (opposite sensors)
- **F31, F32, F33** = Faulty devices spike to 500-1000!
- **PCA showed 37% variance in 2D** → need non-linear models
- **Ensemble beats single model** by combining different mistakes

---

## 🏗️ Pipeline Architecture
```
Raw Data (43,776 rows)
    ↓
Remove 738 Duplicates → 43,038 rows
    ↓
Feature Engineering → 47 to 62 features
    ↓
RobustScaler (handles outliers)
    ↓
Train 5 Models individually
    ↓
Optuna Hyperparameter Tuning
    ↓
Soft Voting Ensemble (4 models)
    ↓
Final Predictions → FINAL.csv (10,944 rows)
```

---

## 📈 Final Results
```
Validation Accuracy : 98.84%
Validation F1 Score : 98.84%
AUC Score           : 99.93%

Confusion Matrix:
True Normal  : 5,120/5,146 (99.49%)
True Faulty  : 3,388/3,462 (97.86%)
False Alarms : 26 only
Missed Faults: 74 only
```

---

## 🛠️ Libraries Used
```
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost, lightgbm
catboost, optuna
```
```

---



