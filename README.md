
#  Road Accident Risk Prediction (Kaggle Top 20%)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-2.x-blue?style=for-the-badge&logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-1.x-green?style=for-the-badge&logo=xgboost" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-3.x-purple?style=for-the-badge&logo=lightgbm" alt="LightGBM">
  <img src="https://img.shields.io/badge/CatBoost-1.x-black?style=for-the-badge&logo=catboost" alt="CatBoost">
</p>

---

## 1. Project Goal and Final Result

This project was developed for a **Kaggle regression competition** to predict a continuous `accident_risk` score (ranging from 0 to 1) based on **road, weather, and traffic conditions**.

Through feature engineering, model comparison, and ensemble learning, the project achieved:

> **Final Score (MAE):** `0.05581`  
> **Leaderboard Rank:** `770 / 4,083 (Top 20%)`

---

##  2. Methodology & Workflow

This project followed an **iterative ML workflow**, from exploratory data analysis to feature engineering and ensembling.

### **Step 1: Data Loading & Exploration**

* Loaded and explored `train.csv` and `test.csv` with **Pandas**.
* Used `.info()`, `.describe()`, and `.corr()` for data understanding.
* Checked missing values and categorical distributions.

---

### **Step 2: Feature Engineering & Selection**

Feature creation and selection were key performance drivers.

* **One-Hot Encoding:**
  Converted categorical features (`lighting`, `weather`, `time_of_day`, `road_type`) using `pd.get_dummies()`.

* **Feature Interaction:**
  Found that road curvature risk depends on the number of lanes:

  ```python
  df["curvature_lane_interaction"] = df["curvature"] / df["num_lanes"]
  ```

* **Feature Selection:**
  Dropped `id`, `road_signs_present`, and `school_season` due to low correlation with target.

---

### **Step 3: Base Model Training**

Two strong gradient boosting models were selected as base learners:

* **XGBoost (`XGBRegressor`)** — Baseline model with `max_depth=7`, `learning_rate=0.1`
* **LightGBM (`LGBMRegressor`)** — Tuned model with `num_leaves=141`

Tree-based models were chosen for their robustness and ability to handle unscaled data.

---

### **Step 4: Advanced Optimization & Ensembling**

Two ensemble strategies were tested:

1. **Simple Averaging**

   ```python
   final_preds = (xgb_preds.clip(0, 1) + lgbm_preds.clip(0, 1)) / 2
   ```

2. **Stacking (Meta-Ensemble)**

   * Used `cross_val_predict` to create out-of-fold predictions.
   * Trained a meta-model (`XGBRegressor`) on base model outputs.

3. **Bayesian Optimization**
   Used `bayes_opt` to fine-tune LightGBM hyperparameters — confirming near-optimal initial values.

---

##  3. Final Submission

The **final model** was built by:

1. Training XGBoost and LightGBM on the **entire training dataset**.
2. Applying identical preprocessing to `test.csv`.
3. Averaging both predictions for submission.

```python
submission["accident_risk"] = (xgb_preds + lgbm_preds) / 2
submission.to_csv("final_submission.csv", index=False)
```

Final Public Leaderboard MAE: **`0.05581`**

---

##  4. Technologies Used

| Category             | Tools / Libraries                                 |
| -------------------- | ------------------------------------------------- |
| **Data Handling**    | `Pandas`, `NumPy`                                 |
| **Visualization**    | `Matplotlib`, `Seaborn`                           |
| **Machine Learning** | `Scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost` |
| **Optimization**     | `bayes_opt` (Bayesian Optimization)               |
| **Environment**      | Jupyter Notebook / Python 3.10+                   |

---

##  5. How to Run Locally

Follow these steps to run the project on your local system.

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/DarkSire7/road_accident_prediction.git
cd road_accident_prediction
```

### **Step 2: Create and Activate a Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On macOS/Linux
```

### **Step 3: Install Dependencies**

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost bayesian-optimization matplotlib seaborn
```

### **Step 4: Run the Notebook or Script After downloading dataset**
Download train.csv from [Kaggle – Road Accident Risk Prediction](https://www.kaggle.com/competitions/playground-series-s5e10/) and place it in data folder 
```bash
jupyter notebook notebooks/accident_ml.ipynb
```

---

##  6. Repository Structure

```
 Road-Accident-Risk-Prediction/
├── data/
│   └── test.csv
├── notebooks/
│   └── accident_ml.ipynb
└── README.md
```

---

##  7. Key Learnings

* Effective **feature engineering** often outperforms deep parameter tuning.
* **Ensemble averaging** is a reliable way to reduce model variance.
* Gained hands-on experience with **Bayesian Optimization**.
* Learned how **tree-based models** handle non-linear patterns effectively.

---

##  8. Acknowledgements

* **Dataset:** [Kaggle – Road Accident Risk Prediction](https://www.kaggle.com/competitions/playground-series-s5e10/)
* **Tools:** Python, Google Colab, Kaggle Notebooks
* **Inspiration:** Kaggle discussions and ensemble learning strategies
