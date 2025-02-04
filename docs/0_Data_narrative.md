# Data Narrative

## 1. Dataset Overview
The dataset comes from the **Titanic: Machine Learning from Disaster** Kaggle competition. It includes passenger details to predict survival based on various features.

## 2. Data Files
- **train.csv**: Includes labeled data (used for training).
- **test.csv**: Includes unlabeled data (used for final predictions).
- **gender_submission.csv**: Example submission file.

## 3. Columns & Metadata
| Column Name    | Description | Data Type | Missing Values? |
|---------------|------------|------------|----------------|
| **PassengerId** | Unique ID of the passenger | Integer | No |
| **Survived** | Target variable (1 = Survived, 0 = Did not survive) | Integer | No (Train only) |
| **Pclass** | Ticket class (1st, 2nd, 3rd) | Integer (Ordinal) | No |
| **Name** | Passenger’s full name | String | No |
| **Sex** | Passenger’s gender | String | No |
| **Age** | Passenger’s age in years | Float | Yes |
| **SibSp** | Number of siblings/spouses aboard | Integer | No |
| **Parch** | Number of parents/children aboard | Integer | No |
| **Ticket** | Ticket number (potentially useful for grouping) | String | No |
| **Fare** | Ticket fare price | Float | No |
| **Cabin** | Cabin number (some passengers share cabins) | String | Yes (Many missing) |
| **Embarked** | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | String (Categorical) | Yes |

## 4. Initial Observations
### **Target Variable**
- The **Survived** column is only available in the training dataset.

### **Potentially Useful Features**
- **Pclass**: Strongly correlated with survival rate.
- **Sex**: Historically, women had a higher survival rate.
- **Family Size (SibSp + Parch)**: Can determine if traveling alone influences survival.
- **Cabin & Ticket**: Can reveal group survival patterns.

### **Challenges**
- **Missing Data**: Age and Cabin have missing values.
- **Feature Engineering**: Extract useful information from names, tickets, and cabin numbers.