# 📌 Exploratory Data Analysis (EDA) - Titanic Dataset

## **1. Introduction**
This document provides a step-by-step analysis of the **Titanic dataset**, examining data distributions, missing values, correlations, and survival patterns.

---

## **2. Dataset Overview**
### **📌 Step 1: Load Data**
- **Command Used:** `load_data()`
- **Findings:**
  - The dataset contains **891 rows** and **12 columns**.
  - Some columns have **missing values**, especially:
    - **Age**: 177 missing values.
    - **Cabin**: 687 missing values (most missing).
    - **Embarked**: 2 missing values.
  - Data types are correct for most columns.

---

## **3. Data Quality Assessment**
### **📌 Step 2: Checking Missing Values**
- **Command Used:** `summarize_data(df)`
- **Findings:**
  - `Age`, `Cabin`, and `Embarked` need **imputation or handling.**
  - `Cabin` has too many missing values, meaning it may **not be useful** for modeling.

---

## **4. Data Distributions**

Before diving into the distributions of numerical and categorical features, we **enhanced our dataset** with two new features:
- **Title** → Extracted from the "Name" column, representing social titles (e.g., Mr., Mrs., Miss).
- **FamilySize** → Calculated as **SibSp + Parch + 1**, representing the total family size of a passenger.

We apply these transformations **before** exploring the feature distributions to ensure a **complete dataset view**.


### **📌 Step 3: Numeric Feature Distributions**
- **Command Used:** `plot_numeric_distributions(df, bins=30)`
- **Findings:**
  - **Age:** Right-skewed with a peak around **30 years old**. Some **missing values** need imputation.
    - **Children (`<10 years old`) had a higher survival rate (~60%)**.
    - **Elderly (`>60 years old`) had the lowest survival rate (~25%)**.
  - **Fare:** Extreme **outliers** suggest a few passengers paid **very high ticket prices**. Most fares are below **$50**, but a few reach **over $500**.
    - **Passengers paying higher fares had a significantly higher survival rate (~70%)**.
  - **Pclass (Passenger Class) - Imbalanced:**
    - **3rd class had the most passengers (~55%)**, but also the **lowest survival rate (~25%)**.
    - **1st class had a survival rate of ~63%**, confirming social class impact.
  - **SibSp (Number of Siblings/Spouses Aboard):** 
    - Majority have **0 siblings/spouses** (traveling alone).
    - A few passengers have **large family groups**.
  - **Parch (Number of Parents/Children Aboard):**
    - Most passengers have **0 parents/children onboard**.
    - Few passengers have **1-3 family members**.
    - **Survival rate was highest for those traveling with 1-3 relatives (~55%)**.
    - **Very large families (`SibSp + Parch > 6`) had a survival rate below 20%**.
  - **FamilySize (NEW):** 
    - Majority of passengers had a **family size of 1 or 2**.
    - Large families (`FamilySize > 6`) were rare.
    - **Small families (2-4 members) had the highest survival rate (~55-60%)**.
    - **Passengers traveling alone (`FamilySize=1`) had a much lower survival rate (~30%)**.
  - **Fare vs. Pclass:**
    - **Higher-class passengers paid significantly more**.
    - **Extreme outliers exist**, meaning some passengers paid **far above the average fare**.
    
### **📌 Step 4: Categorical Feature Distributions**
- **Command Used:** `plot_categorical_distributions(df, layout="single", top_n=100)` and `plot_categorical_distributions(df, selected_columns=["Pclass", "Sex", "Embarked"], hue_feature="Survived", layout="single")`
- **Findings:**
  - **Sex:**
    - **More males (577) than females (314)** were on board (~65% male passengers).
    - **Females had a significantly higher survival rate (~74%)**, while **males had a survival rate of only ~19%**.
  - **Embarked (Port of Embarkation):**
    - Most passengers **embarked from Southampton (S, ~72%)**, but **Cherbourg (C) had a higher survival rate (~55%)**.
    - Suggests **higher-class passengers embarked from Cherbourg**.
  - **Cabin:**
    - **687 missing values**.
    - Some cabins were shared (multiple passengers assigned to the same cabin).
    - Most values are **missing (~77%)**, suggesting it might not be directly useful.
    - **Passengers with recorded cabins had a much higher survival rate (~67%)**.
  - **Ticket:**
    - **Highly unique** (many passengers had different ticket numbers).
    - **Unlikely to be useful for predictive modeling**.
    - **Some shared ticket numbers had better survival rates**, indicating possible group advantages.
  - **Title (NEW):**
    - The most common titles were **Mr (57%), Miss (19%), Mrs (15%), and Master (4%)**.
    - **Highest survival rates:** "Miss" (~70%), "Mrs" (~79%), and "Master" (~57%) (indicating priority for women and children).
    - **Lowest survival rates:** "Mr" (~16%), "Rev" (0%), and "Capt" (0%).
    - **Rare titles (Dr., Sir, Lady) were grouped as ‘Rare’**.
    - **Title had a strong correlation with survival**, as certain titles (e.g., "Master" for young boys) had higher survival rates.

---

### **📌 Key Takeaways from Feature Distributions and Survival Analysis**
✅ **Strongest survival predictors:** `Pclass`, `Sex`, `Fare`, `FamilySize`, and `Title`.  
✅ **Being female, a child (`Age < 10`), or from 1st class significantly increased survival chances.**  
✅ **Traveling alone (`FamilySize=1`) or being male (`Sex=male`) drastically reduced survival chances.**  
✅ **Passengers from Cherbourg had better survival rates, possibly due to higher-class passengers.**  

---

## **5. Correlation Analysis**
### **📌 Step 5: Correlation Heatmap**
- **Command Used:** `plot_correlation_heatmap(df, show_dataframe=True)`
- **Findings:**
  - **Survival Correlations (`Survived` column):**
    - **Pclass (-0.338)** → **Strong negative correlation** with survival:
      - **1st class passengers had a higher survival rate**, while **3rd class passengers had a significantly lower survival rate**.
    - **Fare (+0.257)** → **Passengers who paid higher fares had a higher survival rate**.
    - **Parch (+0.081)** → **Passengers traveling with parents/children had a slightly better survival rate**.
    - **SibSp (-0.035)** → **Little to no impact on survival**.
    - **Age (-0.077)** → **Weak negative correlation**, meaning **younger passengers had slightly better survival chances**, but not significantly.

  - **Key Feature Correlations:**
    - **Pclass & Fare (-0.549)** → **Higher-class passengers paid significantly higher fares**.
    - **Pclass & Age (-0.369)** → **1st class passengers were generally older**, while **3rd class passengers were younger**.
    - **SibSp & Parch (+0.414)** → **Passengers with siblings often traveled with parents/children**.
    - **SibSp & Age (-0.308)** → **Younger passengers were more likely to travel with siblings/spouses**.
    - **Fare & Parch (+0.216)** → **Passengers with families (more Parch) tended to pay higher fares**.

  - **Weak or No Correlations:**
    - **PassengerId (-0.005)** → **No impact on survival**.
    - **SibSp (-0.035) & Parch (0.081)** → **Minor influence on survival**.
    - **Age (-0.077)** → **Weaker than expected correlation with survival**.

  - **Key Takeaways from Correlation Matrix:**
    ✅ **Most Important Survival Factors** → `Pclass` and `Fare` remain the strongest predictors of survival.  
    ✅ **Weak Effect of `SibSp` and `Parch`** → However, they could still be useful when combined into a `FamilySize` feature.  
    ✅ **Higher Class = Older Passengers** → Feature engineering might benefit from analyzing **age-based survival patterns within each class**.

---

## **6. Survival Analysis**
### **📌 Step 6: Survival by Key Features**
- **Command Used:** `plot_survival_analysis(df)`
- **Findings:**
  - **Pclass**: 1st class passengers had the highest survival rate.
  - **Sex**: Females had a significantly higher survival rate.
  - **Embarked**: Passengers from **Cherbourg (C)** had the highest survival rate.

### **📌 Step 7: Survival by Title**
- **Command Used:** `extract_title(df)` and `plot_survival_analysis(df, features=["Title"])`
- **Findings:**
  - **"Master" and "Miss" had higher survival rates** (children prioritized for rescue).
  - **"Mr." had the lowest survival rate.**

### **📌 Step 8: Survival by Family Size**
- **Command Used:** `plot_survival_analysis(df, features=["FamilySize"])`
- **Findings:**
  - **Single travelers had lower survival rates**.
  - **Small families (2-4 members) had the highest survival rates**.
  - **Very large families had lower survival rates** (possible evacuation difficulties).

---

## **7. Key Takeaways**
✅ **Most important survival factors:**
   - **Gender**: Females had **much higher survival** than males.
   - **Class**: Higher class = **higher survival rates**.
   - **Embarked**: Cherbourg (C) had the highest survival rate.
   - **Title**: “Mr.” had the lowest survival rate, while children had better survival.
   - **Family Size**: Small families survived more often than singles or large families.

📌 **Next Steps:**
   - **Feature engineering**: Create new meaningful features based on findings.
   - **Data cleaning**: Handle missing values in `Age`, `Embarked`, and `Cabin`.
   - **Build Machine Learning models** based on insights.

---
