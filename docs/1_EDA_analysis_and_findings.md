# 📌 Exploratory Data Analysis (EDA) - Titanic Dataset

## **1. Introduction**
This document provides a step-by-step analysis of the **Titanic dataset**, examining data distributions, missing values, correlations, and survival patterns.

---

## **2. Dataset Overview**
### **📌 Step 1: Load Data and Data Quality Assessment**
- **Command Used:** 
```py
load_data()
summarize_data(df)
```
- **Findings:**
  - The dataset contains **891 rows** and **12 columns**.
  - Some columns have **missing values**, need **imputation or handling.**:
    - **`Age`**: 177 missing values.
    - **`Cabin`**: 687 missing values (most missing).
    - **`Embarked`**: 2 missing values.
  - `Cabin` has too many missing values, meaning it may **not be useful** for modeling.
  - Data types are correct for most columns.
  - No duplicates found.

---

## **3. Data Distributions**
### **📌 Step 2: Add new features**
- **Command Used:** 
```py
df = extract_title(df)
df = create_family_size(df)
``` 
Before diving into the distributions of numerical and categorical features, we **enhanced our dataset** with two new features:
- **`Title`** → Extracted from the "Name" column, representing social titles (e.g., Mr., Mrs., Miss).
- **`FamilySize`** → Calculated as **`SibSp + Parch + 1`**, representing the total family size of a passenger.

We apply these transformations **before** exploring the feature distributions to ensure a **complete dataset view**.

### **📌 Step 3: Numeric Feature Distributions**
- **Command Used:** 
```py
plot_distributions(df, layout="single")
plot_numeric_distributions(df, layout="single", selected_columns=["Age", "Fare", "FamilySize", "Pclass"], hue_feature="Survived", bins=10)
``` 
- **Findings:**
  - **`Age`:** Right-skewed with a peak around **30 years old**. Some **missing values** need imputation.
    - **Children (`<10 years old`) had a higher survival rate (~60%)**.
    - **Elderly (`>60 years old`) had the lowest survival rate (~25%)**.
  - **`Fare`:** Extreme **outliers** suggest a few passengers paid **very high ticket prices**. Most fares are below **$50**, but a few reach **over $500**.
    - **Passengers paying higher fares had a significantly higher survival rate (~70%)**.
  - **`Pclass` (Passenger Class) - Imbalanced:**
    - **3rd class had the most passengers (~55%)**, but also the **lowest survival rate (~25%)**.
    - **1st class had a survival rate of ~63%**, confirming social class impact.
  - **`SibSp` (Number of Siblings/Spouses Aboard):** 
    - Majority have **0 siblings/spouses** (traveling alone).
    - A few passengers have **large family groups**.
  - **`Parch` (Number of Parents/Children Aboard):**
    - Most passengers have **0 parents/children onboard**.
    - Few passengers have **1-3 family members**.
    - **Survival rate was highest for those traveling with 1-3 relatives (~55%)**.
    - **Very large families (`SibSp + Parch > 6`) had a survival rate below 20%**.
  - **`FamilySize` (NEW):** 
    - Majority of passengers had a **family size of 1 or 2**.
    - Large families (`FamilySize > 6`) were rare.
    - **Small families (2-4 members) had the highest survival rate (~55-60%)**.
    - **Passengers traveling alone (`FamilySize=1`) had a much lower survival rate (~30%)**.
  - **`Fare` vs. `Pclass`:**
    - **Higher-class passengers paid significantly more**.
    - **Extreme outliers exist**, meaning some passengers paid **far above the average fare**.
    
### **📌 Step 4: Categorical Feature Distributions**
- **Command Used:**
```py
plot_categorical_distributions(df, layout="single", top_n=100)
plot_categorical_distributions(df, selected_columns=["Sex", "Embarked", "Title"], hue_feature="Survived", layout="single")
``` 
- **Findings:**
  - **`Sex`:**
    - **More males (577) than females (314)** were on board (~65% male passengers).
    - **Females had a significantly higher survival rate (~74%)**, while **males had a survival rate of only ~19%**.
  - **`Embarked` (Port of Embarkation):**
    - Most passengers **embarked from Southampton (S, ~72%)**, but **Cherbourg (C) had a higher survival rate (~55%)**.
    - Suggests **higher-class passengers embarked from Cherbourg**.
  - **`Cabin`:**
    - **687 missing values**.
    - Some cabins were shared (multiple passengers assigned to the same cabin).
    - Most values are **missing (~77%)**, suggesting it might not be directly useful.
    - **Passengers with recorded cabins had a much higher survival rate (~67%)**.
  - **`Ticket`:**
    - **Highly unique** (many passengers had different ticket numbers).
    - **Unlikely to be useful for predictive modeling**.
    - **Some shared ticket numbers had better survival rates**, indicating possible group advantages.
  - **`Title` (NEW):**
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

## **4. Correlation Analysis**
### **📌 Step 5: Correlation Matrix**
- **Command Used:**
```py
plot_correlation_heatmap(df, show_dataframe=True)
plot_correlation_heatmap(df, include_categorical=True, show_dataframe=True)
``` 
We analyze correlations using **Pearson’s method** by default, but **Spearman** and **Kendall** methods can also be applied.

### ** Steps Taken**
  1. **Encoded categorical variables** (`Title`, `Embarked`, etc.) to include them in the correlation matrix.
  2. **Generated a correlation heatmap** to visualize feature relationships.
  3. **Displayed the correlation matrix as a DataFrame** for detailed numerical analysis.
- **Findings:**
  - **Survival Correlations (`Survived` column):**
    - **`Pclass` (-0.338)** → **Strong negative correlation** with survival:
      - **1st class passengers had a higher survival rate**, while **3rd class passengers had a significantly lower survival rate**.
    - **`Fare` (+0.257)** → **Shows a moderate positive correlation with survival, indicating that passengers who paid higher fares had a better chance of survival.**.
    - **`Parch` (+0.081)** → **Passengers traveling with parents/children had a slightly better survival rate**.
    - **`SibSp` (-0.035)** → **Little to no impact on survival**.
    - **`Age` (-0.077)** → **Weak negative correlation**, meaning **younger passengers had slightly better survival chances**, but not significantly.
    - **`Sex` (-0.543)** → **is strongly correlated, showing that female passengers had a significantly higher survival rate**.
    - **`Title` (-0.071)** → **Weak correlation with survival. Which means that `Title` does not have a strong direct impact on survival. This suggests that while some titles (e.g., "Master" for young boys, "Mrs" for married women) may indicate survival trends, they are not strongly predictive on their own. Instead, `Title` may still be useful in combination with other features like `Sex` and `Age`.**.
    - **`Embarked` (-0.163)** → **Weak to low correlation with survival. Suggests that embarkation had some influence, but not strongly. Certain ports of embarkation had different passenger compositions (e.g., more first-class passengers at some ports), which indirectly influenced survival rates. However, this effect is likely a secondary relationship rather than a direct causal factor.**.

  - **Key Feature Correlations:**
    - **`Pclass` & `Fare` (-0.549)** → **Higher-class passengers paid significantly higher fares**.
    - **`Pclass` & `Age` (-0.369)** → **1st class passengers were generally older**, while **3rd class passengers were younger**.
    - **`SibSp` & `Parch` (+0.414)** → **Passengers with siblings often traveled with parents/children**.
    - **`SibSp` & `Age` (-0.308)** → **Younger passengers were more likely to travel with siblings/spouses**.
    - **`FamilySize` (~0.89 with `SibSp`, ~0.78 with `Parch`)** → **Expected since `FamilySize` is derived from `SibSp` and `Parch`.**
    - **`Fare` & `Parch` (+0.216)** → **Passengers with families (more Parch) tended to pay higher fares**.

  - **Key Takeaways from Correlation Matrix:**
  
    ✅ **Most Important Survival Factors** → `Pclass` and `Fare` remain the strongest predictors of survival.  
    ✅ **Weak Effect of `SibSp` and `Parch`** → However, they could still be useful when combined into a `FamilySize` feature.  
    ✅ **Higher Class = Older Passengers** → Feature engineering might benefit from analyzing **age-based survival patterns within each class**.

# 📌 Correlation Analysis Breakdown

Below is a breakdown of correlations, categorized by their strength.

---

## 🔴 Strong Correlations (|r| > 0.5)
- **Pclass vs. Cabin (0.68)** → Higher-class passengers had more recorded cabin numbers.
- **SibSp vs. FamilySize (0.89)** → Family size is strongly determined by siblings/spouses.
- **Parch vs. FamilySize (0.78)** → Larger families onboard had more parent/child relations.
- **Title vs. Age (0.51)** → Certain titles are associated with specific age groups (e.g., "Master" for young boys).
- **Fare vs. Pclass (-0.55)** → First-class passengers paid significantly higher fares.
- **Sex vs. Survived (-0.54)** → Strong evidence that **females survived at a much higher rate**.

---

## 🟡 Moderate Correlations (0.3 ≤ |r| < 0.5)
- **Survived vs. Fare (0.26)** → Passengers who paid higher fares had a greater chance of survival.
- **Survived vs. Cabin (-0.25)** → Assigned cabins had some impact on survival.
- **Pclass vs. Age (-0.36)** → Older passengers were more likely to be in first class.
- **Pclass vs. Survived (-0.34)** → Higher-class passengers had a higher survival rate.
- **Age vs. SibSp (-0.30)** → Younger passengers tended to have siblings aboard.
- **Fare vs. FamilySize (0.21)** → Larger families often paid higher combined fares.
- **Embarked vs. Fare (0.19)** → Some embarkation points were associated with higher ticket prices.

---

## 🟢 Low Correlations (0.1 ≤ |r| < 0.3)
- **Survived vs. Embarked (-0.16)** → Embarkation point had minor influence on survival.
- **Survived vs. Title (-0.07)** → Titles reflect some survival disparity, but not strongly.
- **Title vs. FamilySize (-0.20)** → Some titles are associated with family roles (e.g., "Mrs" likely to have dependents).
- **Fare vs. Age (0.09)** → Older passengers tended to pay slightly higher fares.
- **Pclass vs. Embarked (0.15)** → Some embarkation points had more first-class passengers.

---

## ⚪ Weak or No Correlation (|r| < 0.1)
- **PassengerId vs. Everything** → As expected, PassengerId has no meaningful correlation.
- **Survived vs. SibSp (-0.03)** → Weak relationship between siblings and survival.
- **Survived vs. Parch (0.08)** → Small positive correlation; families slightly increased survival.
- **Ticket, Name, and Embarked** → These features show little correlation with survival or other variables.

---

## 📌 Summary
- **Sex and Fare show the strongest survival impact**.
- **Pclass and Cabin are highly related** to fare and social class.
- **Family-based features (SibSp, Parch, FamilySize) are highly correlated**.
- **Embarked and Title have minor but noticeable relationships**.

---

## 🚀 Next Steps
1. Consider using **Sex, Fare, Pclass, Title, and FamilySize** as key predictors.
2. Remove highly correlated redundant features (e.g., drop **SibSp/Parch** in favor of **FamilySize**).
3. Engineer new features using strong relationships (e.g., combining **Cabin and Pclass**).