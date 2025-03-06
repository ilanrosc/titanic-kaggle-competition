# **2. Preprocessing Documentation**

## **Overview**
This document outlines the preprocessing steps applied to the Titanic dataset, including handling missing values, feature engineering, and preparing the data for model training.

## **1. Data Loading**
- Loaded the preprocessed dataset saved in `data/processed/train_eda.csv`.
- Ensured that all previous EDA-derived columns such as `Title` and `FamilySize` are present.

## **2. Handling Missing Values**

### **Age Column:**
- Applied a hybrid imputation strategy:
   - Used `median` imputation for a simple and effective handling.
   - Plan to explore predictive imputation based on similar passengers' features in the next steps.

### **Cabin Column:**
- Created a binary indicator column `Cabin_Missing` to capture missingness as a potential feature.
- Dropped the original `Cabin` column due to a high missing rate (~77%).
- Reasoning:
   - Missingness may correlate with passenger class and survival.
   - Binary indicator preserves useful information without the sparsity of the original column.

### **Embarked Column:**
- Used `mode` imputation to fill 2 missing values in `Embarked`.
- Reasoning:
   - Simple and effective due to a very low count of missing values.   