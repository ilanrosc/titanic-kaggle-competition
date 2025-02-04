# 🚀 Scripts Overview

## **📂 Project Structure**
This project follows a modular structure to ensure flexibility, reusability, and maintainability. Below is an overview of the scripts and their responsibilities.

---

## **1️⃣ Data Handling**
### 📌 `data_loader.py`
- **Purpose:** Loads data from CSV files into Pandas DataFrames.
- **Functions:**
  - `load_data(data_path="data", filename="train.csv")` → Loads and returns a dataset.

### 📌 `data_split.py`
- **Purpose:** Splits datasets into **train-test** sets.
- **Functions:**
  - `split_data(df, target_column, test_size=0.2, random_state=42, exclude_columns=None)`
  - Handles missing columns safely.

---

## **2️⃣ Exploratory Data Analysis (EDA)**
### 📌 `eda.py`
- **Purpose:** Analyzes the dataset structure and feature distributions.
- **Functions:**
  - `summarize_data(df)` → Displays dataset information, missing values, and summary statistics.
  - `plot_distributions(df, exclude_columns=None)` → Plots histograms for numerical features.

---

## **3️⃣ Feature Engineering**
### 📌 `feature_engineering.py`
- **Purpose:** Extracts additional features to improve model performance.
- **Functions:**
  - `extract_title(df, name_column="Name")` → Extracts passenger titles (e.g., Mr., Miss, Dr.).
  - `create_family_size(df, sibsp_col="SibSp", parch_col="Parch")` → Computes family size.

---

## **4️⃣ Data Preprocessing**
### 📌 `preprocessing.py`
- **Purpose:** Prepares data for machine learning.
- **Functions:**
  - `fill_missing_values(df, strategy="median", constant_value=None)` → Handles missing values.
  - `encode_categorical(df)` → Converts categorical values into numeric format.
  - `scale_features(df)` → Standardizes numerical features.

---

## **5️⃣ Model Training & Evaluation**
### 📌 `model_training.py`
- **Purpose:** Trains and evaluates a **Random Forest classifier**.
- **Functions:**
  - `train_model(df, target_column="Survived", test_size=0.2, exclude_columns=None)`
  - Returns trained model and accuracy.

### 📌 `model_stacking.py`
- **Purpose:** Implements **model stacking** (Random Forest + Gradient Boosting).
- **Functions:**
  - `stack_models(df)` → Averages predictions from two classifiers.

### 📌 `evaluation.py`
- **Purpose:** Computes model performance metrics.
- **Functions:**
  - `evaluate_model(y_true, y_pred)` → Returns **accuracy, F1-score, and ROC AUC**.

---

## **6️⃣ Kaggle Submission**
### 📌 `submission.py`
- **Purpose:** Generates a submission file for Kaggle.
- **Functions:**
  - `generate_submission(model, test_data, filename="submission.csv")` → Saves predictions.

---
