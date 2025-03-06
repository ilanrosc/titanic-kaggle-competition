from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.special import logit
from scipy.stats import shapiro, normaltest, skew, kurtosis
from .data_loader import load_data

def fill_missing_values(df: pd.DataFrame, 
                        num_strategy: str = "median", 
                        cat_strategy: str = "mode", 
                        fill_value=None,
                        columns: list = None, 
                        exclude_columns=None) -> pd.DataFrame:
    """
    Handles missing values separately for numerical and categorical features.

    Args:
        df (pd.DataFrame): The dataset.
        num_strategy (str): Strategy for numerical columns - "mean", "median", "mode", "constant", "ffill", "bfill", "drop", or "predictive".
        cat_strategy (str): Strategy for categorical columns - "mode", "constant", "ffill", "bfill", "drop", or "predictive".
        fill_value: Value to use when strategy="constant".
        columns (list, optional): Specific columns to fill. Fills all columns if None.
        exclude_columns (list, optional): List of columns to exclude from processing.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """

    df_copy = df.copy()

    # Handle specified columns
    if columns:
        cols_to_process = columns
    else:
        cols_to_process = df_copy.columns

    # Exclude specified columns
    if exclude_columns:
        cols_to_process = [col for col in cols_to_process if col not in exclude_columns]

    # Identify numerical and categorical columns
    num_cols = df_copy[cols_to_process].select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df_copy[cols_to_process].select_dtypes(include=["object", "category"]).columns

    # Handle missing values for numerical columns
    for col in num_cols:
        if df_copy[col].isnull().sum() > 0:
            if num_strategy == "mean":
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif num_strategy == "median":
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif num_strategy == "mode":
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif num_strategy == "constant":
                if fill_value is not None:
                    df_copy[col] = df_copy[col].fillna(fill_value)
                else:
                    raise ValueError("For strategy='constant', a fill_value must be provided.")
            elif num_strategy == "ffill":
                df_copy[col] = df_copy[col].ffill()
            elif num_strategy == "bfill":
                df_copy[col] = df_copy[col].bfill()
            elif num_strategy == "drop":
                df_copy.dropna(subset=[col], inplace=True)
            elif num_strategy == "predictive":
                # Separate missing and non-missing data
                missing_df = df_copy[df_copy[col].isna()]
                non_missing_df = df_copy[~df_copy[col].isna()]

                # Select features for prediction
                features = non_missing_df.drop(columns=[col]).columns

                # Choose model based on column type
                model = RandomForestRegressor(n_estimators=100, random_state=42)

                # Train model
                model.fit(non_missing_df[features], non_missing_df[col])

                # Predict missing values
                predicted_values = model.predict(missing_df[features])
                df_copy.loc[df_copy[col].isna(), col] = predicted_values

    # Handle missing values for categorical columns
    for col in cat_cols:
        if df_copy[col].isnull().sum() > 0:
            if cat_strategy == "mode":
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif cat_strategy == "constant":
                if fill_value is not None:
                    df_copy[col] = df_copy[col].fillna(fill_value)
                else:
                    raise ValueError("For strategy='constant', a fill_value must be provided.")
            elif cat_strategy == "ffill":
                df_copy[col] = df_copy[col].ffill()
            elif cat_strategy == "bfill":
                df_copy[col] = df_copy[col].bfill()
            elif cat_strategy == "drop":
                df_copy.dropna(subset=[col], inplace=True)
            elif cat_strategy == "predictive":
                # Separate missing and non-missing data
                missing_df = df_copy[df_copy[col].isna()]
                non_missing_df = df_copy[~df_copy[col].isna()]

                # Select features for prediction
                features = non_missing_df.drop(columns=[col]).columns

                # Encode categorical features
                encoders = {}
                for feature in features:
                    if df_copy[feature].dtype == "object":
                        encoders[feature] = LabelEncoder()
                        df_copy[feature] = encoders[feature].fit_transform(df_copy[feature].astype(str))

                # Choose model based on column type
                model = RandomForestClassifier(n_estimators=100, random_state=42)

                # Train model
                model.fit(non_missing_df[features], non_missing_df[col])

                # Predict missing values
                predicted_values = model.predict(missing_df[features])
                df_copy.loc[df_copy[col].isna(), col] = predicted_values

                # Decode categorical features
                for feature in features:
                    if feature in encoders:
                        df_copy[feature] = encoders[feature].inverse_transform(df_copy[feature].astype(int))

    return df_copy

def encode_categorical(df):
    """Encodes categorical features dynamically."""
    df = df.copy() 
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def scale_features(df):
    """Standardizes numerical features dynamically."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = [col for col in num_cols if col.lower() not in ["passengerid", "survived"]]  # Exclude IDs & target
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])
    return df

def check_normality(df: pd.DataFrame, exclude_columns=None, show: bool = True, layout: str = "single", plot_type: str = "both"):
    """
    Checks if numerical features are normally distributed and suggests an outlier handling method.
    Use Shapiro-Wilk for small samples and D'Agostino's K-squared for larger datasets.

    Args:
        df (pd.DataFrame): The dataset.
        exclude_columns (list, optional): List of columns to exclude from analysis.
        show (bool): If True, displays Histogram and KDE plots for each feature. Default is True.
        layout (str): "multiple" (default) to show one plot per figure or "single" to show all in one figure.
        plot_type (str): "hist" for Histogram & KDE, "qq" for QQ plots, "both" for both types. Default is "hist".

    Returns:
        dict: Summary of normality tests for each column.
    """
    results = {}
    cols_to_process = df.select_dtypes(include=["float64", "int64"]).columns
    if exclude_columns:
        cols_to_process = [col for col in cols_to_process if col not in exclude_columns]

    print("📊 **Normality Check Results:**")
    print("-" * 50)
    for col in cols_to_process:
        # Choose the appropriate test based on sample size
        if len(df[col].dropna()) < 5000:
            p_value = shapiro(df[col].dropna()).pvalue
        else:
            p_value = normaltest(df[col].dropna()).pvalue

        skewness = skew(df[col].dropna())
        kurt = kurtosis(df[col].dropna())

        is_normal = p_value > 0.05
        suggested_method = "zscore" if is_normal else "iqr"

        print(f"🔹 {col}: {'Normal' if is_normal else 'Skewed'} distribution (p-value: {p_value:.4f}, skewness: {skewness:.2f}, kurtosis: {kurt:.2f})")
        print(f"   ➡️ Suggested outlier handling method: {suggested_method.upper()}\n")

        results[col] = {
            "is_normal": is_normal,
            "p_value": p_value,
            "skewness": skewness,
            "kurtosis": kurt,
            "suggested_method": suggested_method
        }

    # Display plots if show is True
    if show:
        if layout == "single":
            num_cols = len(cols_to_process)
            fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(12, 5 * ((num_cols + 1) // 2)))
            axes = axes.flatten()

            for i, col in enumerate(cols_to_process):
                # Histogram with KDE (Kernel Density Estimate)
                if plot_type in ["hist", "both"]:
                    sns.histplot(df[col].dropna(), bins=30, kde=True, color="skyblue", edgecolor="black", alpha=0.6, ax=axes[i])
                    axes[i].set_title(f"Histogram & KDE for {col}")
                # QQ Plot (to check normality visually)
                if plot_type in ["qq", "both"]:
                    stats.probplot(df[col].dropna(), dist="norm", plot=axes[i])
                    axes[i].set_title(f"QQ Plot for {col}")

            if num_cols % 2 != 0:
                fig.delaxes(axes[-1])  # Remove empty subplot if an odd number of variables

            plt.tight_layout()
            plt.show()
            
        else:
            # Show one plot per figure
            for col in cols_to_process:
                if plot_type in ["hist", "both"]:
                    plt.figure(figsize=(10, 5))
                    sns.histplot(df[col].dropna(), bins=30, kde=True, color="skyblue", edgecolor="black", alpha=0.6)
                    plt.title(f"Histogram & KDE Plot for {col} - {'Normal' if results[col]['is_normal'] else 'Skewed'} Distribution")
                    plt.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
                    plt.axvline(df[col].median(), color='green', linestyle='--', label='Median')
                    plt.legend()
                    plt.show()
                if plot_type in ["qq", "both"]:
                    plt.figure(figsize=(8, 6))
                    stats.probplot(df[col].dropna(), dist="norm", plot=plt)
                    plt.title(f"QQ Plot for {col}")
                    plt.grid()
                    plt.show()

    return results

def identify_outliers(df: pd.DataFrame, 
                      method: str = "iqr", 
                      threshold: float = 1.5, 
                      exclude_columns=None, 
                      norm_results=None,
                      show_boxplot: bool = True) -> pd.DataFrame:
    """
    Identifies and counts outliers in numerical features based on the specified method.

    Args:
        df (pd.DataFrame): The dataset.
        method (str): Outlier detection method - "iqr", "zscore", or "percentile".
        threshold (float): Threshold for outlier detection.
        exclude_columns (list, optional): List of columns to exclude from analysis.
        norm_results (dict, optional): Results from check_normality() for suggested method.
        show_boxplot (bool): If True, displays boxplots of features with outliers. Default is True.

    Returns:
        pd.DataFrame: Summary of outlier counts and suggested handling methods.
    """
    df = df.copy()
    summary = []
    cols_to_process = df.select_dtypes(include=["float64", "int64"]).columns
    if exclude_columns:
        cols_to_process = [col for col in cols_to_process if col not in exclude_columns]

    outlier_columns = []  # To store columns with outliers for boxplot

    for col in cols_to_process:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

        elif method == "percentile":
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)

        # Find outliers
        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        total_outliers = len(outlier_indices)
        outlier_percentage = (total_outliers / len(df)) * 100

        if total_outliers > 0:
            outlier_columns.append(col)  # Track columns with outliers

        # Suggested method based on normality results if available
        suggested_method = norm_results[col]["suggested_method"].upper() if norm_results else method.upper()

        # Append results to summary
        summary.append({
            "Feature": col,
            "Total Outliers": total_outliers,
            "Outlier Percentage (%)": round(outlier_percentage, 2),
            "Suggested Method": suggested_method,
            "Method applied": method
        })

    # Convert summary to DataFrame for better readability
    outlier_summary = pd.DataFrame(summary)
    print("📊 **Outlier Summary:**")
    print(outlier_summary)
    
    # Plot boxplots for outliers if show_boxplot is True
    if show_boxplot and outlier_columns:
        fig, axes = plt.subplots(nrows=len(outlier_columns), figsize=(8, 5 * len(outlier_columns)))
        axes = axes if len(outlier_columns) > 1 else [axes]

        for idx, col in enumerate(outlier_columns):
            sns.boxplot(data=df, x=col, ax=axes[idx], color="lightcoral")
            axes[idx].set_title(f"Boxplot of {col} (Outliers Highlighted)")

        plt.tight_layout()
        plt.show()
    return outlier_summary

def compute_skewness(df: pd.DataFrame, exclude_columns=None, show: bool = True, view: str = "multiple"):
    """
    Computes skewness for numerical features and provides interpretation.

    Args:
        df (pd.DataFrame): The dataset.
        exclude_columns (list, optional): List of columns to exclude from skewness analysis.
        show (bool): If True, displays histograms with KDE for each feature. Default is True.
        view (str): "multiple" for separate plots or "single" for all in one figure. Default is "multiple".

    Returns:
        dict: Skewness values for each numerical feature.
    """
    # Select numerical columns
    cols_to_process = df.select_dtypes(include=["float64", "int64"]).columns
    if exclude_columns:
        cols_to_process = [col for col in cols_to_process if col not in exclude_columns]

    skewness_results = {}

    print("📊 **Skewness Analysis:**")
    print("-" * 50)

    # Calculate skewness and print interpretation
    for col in cols_to_process:
        skew_val = skew(df[col].dropna())
        skewness_results[col] = skew_val

        # Interpret skewness level
        if abs(skew_val) < 0.5:
            skew_level = "Approximately symmetric"
        elif 0.5 <= abs(skew_val) < 1:
            skew_level = "Moderately skewed"
        else:
            skew_level = "Highly skewed"

        # Determine skew direction
        if skew_val > 0:
            direction = "Right (Positive) skew"
        elif skew_val < 0:
            direction = "Left (Negative) skew"
        else:
            direction = "Symmetric"

        print(f"🔹 {col}: Skewness = {skew_val:.2f} ({skew_level}, {direction})")


    # Plot histograms with KDE if show is True
    if show:
        if view == "single":
            num_cols = len(cols_to_process)
            fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(12, 5 * ((num_cols + 1) // 2)))
            axes = axes.flatten()

            for i, col in enumerate(cols_to_process):
                sns.histplot(df[col].dropna(), bins=30, kde=True, color="skyblue", edgecolor="black", alpha=0.6, ax=axes[i])
                axes[i].set_title(f"Histogram & KDE for {col} - Skewness: {skewness_results[col]:.2f}")

            if num_cols % 2 != 0:
                fig.delaxes(axes[-1])  # Remove empty subplot if an odd number of variables

            plt.tight_layout()
            plt.show()
        else:
            for col in cols_to_process:
                plt.figure(figsize=(8, 5))
                sns.histplot(df[col].dropna(), bins=30, kde=True, color="skyblue", edgecolor="black", alpha=0.6)
                plt.title(f"Histogram & KDE for {col} - Skewness: {skewness_results[col]:.2f}")
                plt.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
                plt.axvline(df[col].median(), color='green', linestyle='--', label='Median')
                plt.legend()
                plt.show()

    return skewness_results

def transform_feature(df: pd.DataFrame, columns_strategy: dict = None, default_strategy: str = "log") -> pd.DataFrame:
    """
    Applies specified transformations to selected features.

    Args:
        df (pd.DataFrame): The dataset.
        columns_strategy (dict, optional): Dictionary with column names as keys and transformation strategies as values.
        default_strategy (str): Default transformation strategy for columns not specified in columns_strategy.

    Returns:
        pd.DataFrame: Dataset with transformed features.
    """
    # Create a copy to avoid modifying the original dataset
    df = df.copy()
    transformed_features = []

    # If no column-specific strategy is provided, apply default to all numeric columns
    if columns_strategy is None:
        columns_strategy = {col: default_strategy for col in df.select_dtypes(include=["int64", "float64"]).columns}

    for col, strategy in columns_strategy.items():
        try:
            # Apply transformation based on strategy
            if strategy == "log":
                df[col] = np.log1p(df[col].clip(lower=0))  # Clip to handle negative values
            elif strategy == "sqrt":
                df[col] = np.sqrt(df[col].clip(lower=0))
            elif strategy == "cbrt":
                df[col] = np.cbrt(df[col])
            elif strategy == "boxcox":
                df[col], _ = stats.boxcox(df[col].clip(lower=1e-5))  # Clip to avoid zero/negative
            elif strategy == "yeojohnson":
                df[col], _ = stats.yeojohnson(df[col])
            elif strategy == "reciprocal":
                df[col] = 1 / df[col].replace(0, np.nan)  # Replace zero to avoid division by zero
            elif strategy == "rank":
                df[col] = stats.rankdata(df[col])
            elif strategy == "logit":
                df[col] = logit(df[col].clip(1e-5, 1 - 1e-5))  # Clip to avoid 0 and 1
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            transformed_features.append((col, strategy))

        except Exception as e:
            print(f"⚠️ Could not transform {col} using '{strategy}': {e}")

    # Summary of transformations
    print("\n📊 **Transformation Summary:**")
    for col, strategy in transformed_features:
        print(f"   ➡️ {col}: Transformed using '{strategy}' strategy")

    return df


if __name__ == "__main__":
    df = load_data()  
    
    # Example usage:
    # print("Fill missing values using different strategies for numerical & categorical columns")
    # df_cleaned_v1 = fill_missing_values(df, num_strategy="median", cat_strategy="mode", exclude_columns=["PassengerId"])
    # print(df_cleaned_v1.head())
    # print("Forward fill for categorical, mean fill for numerical")
    # df_cleaned_v2 = fill_missing_values(df, num_strategy="mean", cat_strategy="ffill")
    # print(df_cleaned_v2.head())
    # print("Drop missing values in numerical features but use mode for categorical")
    # df_cleaned_v3 = fill_missing_values(df, num_strategy="drop", cat_strategy="mode")
    # print(df_cleaned_v3.head())
    # print("Fill all missing categorical values with 'Unknown' and numeric with median")
    # df_cleaned_v4 = fill_missing_values(df, num_strategy="median", cat_strategy="constant", fill_value="Unknown")
    # print(df_cleaned_v4.head())
    # print("Check normality of all numerical columns")
    # check_normality(df, show=True, layout="multiple", plot_type="both")
    print("Identify outliers using IQR method")
    identify_outliers(df, method="iqr")
    # print("Compute skewness of all numerical columns")
    # compute_skewness(df, exclude_columns=["PassengerId","Survived"], show=True, view="single")
    # print("Transform features using different strategies")
    # df = transform_feature(df, columns_strategy={
    # 'Fare': 'log',
    # 'Age': 'yeojohnson',
    # 'FamilySize': 'sqrt'
    # })