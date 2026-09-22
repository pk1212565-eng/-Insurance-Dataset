# -Insurance-Dataset
# Medical Insurance Charges Prediction - Supervised Learning Pipeline

This repository contains an end-to-end data science workflow designed to clean, process, and analyze a medical insurance dataset. The primary objective is to implement and evaluate multiple **Supervised Learning** algorithms to predict health insurance premium charges and model user profiles.

---

## 📊 Dataset Overview
The dataset contains **1,338 records** of beneficiary data with the following attributes:
*   `age`: Age of primary beneficiary
*   `sex`: Insurance contractor gender (female, male)
*   `bmi`: Body mass index (kg/m²)
*   `children`: Number of dependents covered by health insurance
*   `smoker`: Smoking status (yes, no)
*   `region`: The beneficiary's residential area in the US (northeast, northwest, southeast, southwest)
*   `charges`: Individual medical costs billed by health insurance *(Target Variable)*

---

## ⚙️ Data Preprocessing Pipeline

To prepare the dataset for machine learning models, a robust preprocessing pipeline is implemented:

### 1. Handling Missing Values
*   Identifies and confirms missing structural records.
*   Uses strategies like median imputation for skewed continuous attributes and mode replacement for categorical placeholders if unexpected values occur.

### 2. Encoding Categorical Variables
*   **Binary Attributes:** Column transformations mapping (`yes`/`no`) or (`male`/` female`) values safely to binary bits (`1`/`0`).
*   **Nominal Attributes:** One-Hot Encoding applies dummy expansion on columns like `region` to eliminate arbitrary numeric bias.

### 3. Feature Scaling & Engineering
*   Applies **StandardScaler** to continuously distributed columns (`age`, `bmi`, `children`) to maintain zero mean and unit variance.
*   Ensures gradient-based or distance-dependent algorithms (like SVM and KNN) perform optimally without feature dominance.

---

## 🤖 Implemented Supervised Learning Models

The pipeline trains, tunes, and comprehensively measures the performance of the following models:

### 📈 Regression-Based Models
*   **Linear Regression:** Captures global linear baselines between features like BMI, smoking patterns, and charges.
*   **Logistic Regression:** Evaluates conditional probabilities for profiling data points by classifying binary target constraints.

### 🌲 Tree-Based & Ensemble Models
*   **Decision Trees:** Maps non-linear, step-like splits across attributes for localized predictions.
*   **Random Forest:** Combines bootstrapped collections of deep tree estimators to drastically minimize variance and prevent overfitting.
*   **Gradient Boosting (XGBoost/LightGBM):** Sequentially fits shallow trees to minimize loss residuals, providing high-performance scoring.

### 📐 Distance & Vector-Based Models
*   **K-Nearest Neighbors (KNN):** Implements instance-based spatial neighborhood averaging.
*   **Support Vector Machine (SVM):** Maximizes error boundary margins using structural kernel tricks to handle complex patterns.

---

## 📉 Evaluation Metrics

Models are evaluated using standard metrics to assess accuracy and error rates:
*   **Mean Absolute Error (MAE):** Tells the average magnitude of absolute prediction errors.
*   **Root Mean Squared Error (RMSE):** Heavy penalty metric tracking unexpected outlier variances.
*   **R² Score (Coefficient of Determination):** Shows the percentage of variance explained by model inputs.

---

## 📁 Project Structure

```text
├── insurance_analysis.ipynb   # Complete Jupyter Notebook containing raw execution code
├── README.md                 # Project guide and documentation
└── insurance.csv             # Raw medical billing dataset file
```

---

## 🚀 Quick Start & Installation

1. Clone this repository to your local setup.
2. Install the necessary tracking tools and dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm jupyter
   ```
3. Boot up your workspace engine environment:
   ```bash
   jupyter notebook
   ```
4. Open `insurance_analysis.ipynb` and execute all cells sequentially.
