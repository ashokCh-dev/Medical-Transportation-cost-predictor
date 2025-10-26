# 🏥 Medical Equipments Cost Prediction Challenge

A Machine Learning project developed for the **AIT 511 / Machine Learning** course at IIIT Bangalore.  
This repository implements preprocessing, feature engineering, and multiple regression models to predict medical equipment transportation costs.

---

## 📊 Project Overview

The task is based on the Kaggle competition:  
🔗 [Medical Equipments Cost Prediction Challenge](https://www.kaggle.com/competitions/Medical-Equipments-Cost-Prediction-Challenge)

The dataset contains hospital, supplier, and shipment details such as:
- Equipment dimensions and value  
- Shipping type (urgent, cross-border, fragile)  
- Supplier reliability  
- Order and delivery dates  

The target variable is **`Transport_Cost`** — the total cost of equipment transport.

---

## 🧰 Libraries Used

| Category | Libraries |
|-----------|------------|
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | scikit-learn, xgboost |
| Utilities | joblib, datetime |

---

## 🧼 Data Preprocessing

Preprocessing steps were implemented via a unified **scikit-learn Pipeline** for reproducibility.

1. **Drop Identifiers**  
   - Removed `Hospital_Id`, `Supplier_Name`, and `Hospital_Location`.

2. **Handle Missing Values**  
   - Numeric: median imputation + `RobustScaler`  
   - Categorical: most frequent value or “Unknown”

3. **Date Feature Engineering**
   - Converted `Order_Placed_Date`, `Delivery_Date` → datetime  
   - Derived `delivery_days` = difference in days  
   - Extracted day and month components  
   - Applied **cyclical encoding** using sine/cosine transformations  
   - Added weekend flags

4. **Feature Scaling & Encoding**  
   - Numeric: `RobustScaler`  
   - Categorical: One-Hot Encoding

5. **Pipeline Integration**  
   - Combined all transformations using `ColumnTransformer`  
   - Split data 80/20 for training and validation

---

## 🧠 Models and Tuning

Multiple regression algorithms were trained and tuned via `GridSearchCV`:

| Model | Best Parameters | Kaggle Score |
|--------|----------------|---------------|
| Linear Regression | `fit_intercept=True` | 3,979,238,797 |
| Ridge Regression | `alpha=0.001` | 3,979,238,713 |
| Lasso Regression | `alpha=0.0001` | 3,980,875,734 |
| ElasticNetCV | `alpha=0.0001, l1_ratio=0.1` | 3,979,230,372 |
| Polynomial (deg=2) | `degree=2` | 16,549,063,826 |
| Random Forest | `n_estimators=100, max_depth=8` | 11,653,441,928 |
| AdaBoost | `n_estimators=100, learning_rate=0.2` | 8,913,577,487 |
| XGBoost | `max_depth=3, lr=0.03, n_estimators=100` | 13,903,257,906 |
| Gradient Boosting | `max_depth=4, n_estimators=300` | 23,579,556,424 |

---

## 🏆 Best Model

**Elastic Net Regression**

**Observation:**  
Feature–cost relationships were largely linear, making linear models more effective than ensembles, which overfitted due to limited data.  
The Elastic Net’s hybrid regularization balanced sparsity (L1) and stability (L2), improving generalization.

---

## 📈 Visualizations

The repository includes:
- **Density Plots** for numeric features (`density_<feature>.png`)  
- **Scatter plots** of `Transport_Cost` vs features  
- **Correlation Heatmap** (`heatmap.png`)  

All plots are auto-generated using:
```bash
python plot_density_all.py
