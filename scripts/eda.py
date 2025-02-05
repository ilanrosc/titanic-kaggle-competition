import matplotlib.pyplot as plt
import seaborn as sns
import math
from .data_loader import load_data

def summarize_data(df):
    """Prints dataset info, missing values, and summary statistics."""
    print("📌 Dataset Info:")
    df.info()

    print("\n📌 Missing Values:")
    print(df.isnull().sum())

    duplicate_count = df.duplicated().sum()
    print(f"\n📌 Duplicate Rows: {duplicate_count}")

    print("\n📌 Summary Statistics:")
    print(df.describe())


def plot_distributions(df, exclude_columns=None, layout="multiple"):
    """
    Plots distributions for all numerical columns.

    Args:
        df (pd.DataFrame): Dataset
        exclude_columns (list, optional): List of columns to exclude from visualization.
    """
    exclude_columns = exclude_columns if exclude_columns else []
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    num_cols = [col for col in num_cols if col.lower() not in exclude_columns]  # Exclude specified columns
    
    if layout == "single":
        # Option 1: Display all distributions in one figure
        plt.figure(figsize=(15, 10))
        df[num_cols].hist(bins=30, figsize=(15, 10), edgecolor='black')

        plt.suptitle("Feature Distributions", fontsize=16)
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.show()

    else:
        # Option 2: Display each plot one by one (current behavior)
        for col in num_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

def plot_categorical_distributions(df, exclude_columns=None, selected_columns=None, layout="multiple", top_n=100, hue_feature=None):
    """
    Plots count distributions for categorical features with an optional survival rate split.

    Args:
        df (pd.DataFrame): The dataset.
        exclude_columns (list, optional): List of columns to exclude from visualization.
        selected_columns (list, optional): List of specific categorical columns to plot.
        layout (str): "multiple" (default) to display one-by-one or "single" to show all in one figure.
        top_n (int): Maximum number of unique values to display per category, ordered by count.
        hue_feature (str, optional): Feature by which to check distribution differences (e.g., "Survived").
    """
    
    # Step 1: Identify categorical columns
    exclude_columns = exclude_columns if exclude_columns else []
    cat_cols = df.select_dtypes(include=["object"]).columns
    cat_cols = [col for col in cat_cols if col not in exclude_columns]

    # Step 2: If selected_columns is provided, use only those
    if selected_columns:
        cat_cols = [col for col in cat_cols if col in selected_columns]

    if not cat_cols:
            print("⚠️ No categorical columns found to plot!")
            return
    df_copy = df.copy()

    def add_count_labels(ax):
        """Adds count labels inside bars with auto-color adjustment."""
        for p in ax.patches:
            if p.get_height() > 0:
                # Adjust text position inside the bar
                position_y = p.get_height() / 2  # Centered vertically
                color = "white" if p.get_height() > 10 else "black"  # White text for tall bars, black for short bars
                ax.annotate(f"{int(p.get_height())}", 
                            (p.get_x() + p.get_width() / 2, position_y), 
                            ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    # Step 3: Handle 'single' or 'multiple' layout
    if layout == "single":
        num_cols = len(cat_cols)
        rows = math.ceil(num_cols / 2)

        fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 4))
        axes = axes.flatten()  # Ensure axes are iterable

        for i, col in enumerate(cat_cols):
            top_categories = df_copy[col].value_counts().nlargest(top_n)

            sns.countplot(data=df_copy[df_copy[col].isin(top_categories.index)], 
                          x=col, hue=hue_feature if hue_feature else col, palette="coolwarm", ax=axes[i], 
                          legend=True if hue_feature else False, order=top_categories.index, stat="count")
            axes[i].set_title(f"Distribution of {col}" + (f" by {hue_feature}" if hue_feature else ""))

            # Ensure x-ticks are correctly set before applying labels
            axes[i].set_xticks(range(len(top_categories)))
            axes[i].set_xticklabels(top_categories.index, rotation=30, ha="right")

            # Add count labels inside bars
            add_count_labels(axes[i])

        # Remove empty subplot if num_cols is odd
        if num_cols % 2 != 0:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.show()

    else:
        # Step 4: Plot each category separately
        for col in cat_cols:
            plt.figure(figsize=(8, 5))
            top_categories = df_copy[col].value_counts().nlargest(top_n)

            ax = sns.countplot(data=df_copy[df_copy[col].isin(top_categories.index)], 
                          x=col, hue=hue_feature if hue_feature else col, palette="coolwarm", 
                          legend=True if hue_feature else False, order=top_categories.index, stat="count")
            plt.title(f"Distribution of {col}" + (f" by {hue_feature}" if hue_feature else ""))

            # Ensure x-ticks are correctly set before applying labels
            ax.set_xticks(range(len(top_categories)))
            ax.set_xticklabels(top_categories.index, rotation=30, ha="right")

            # Add count labels inside bars
            add_count_labels(ax)

            plt.show()

def plot_correlation_heatmap(df, selected_columns=None, method="pearson", figsize=(10, 8), annot=True, cmap="coolwarm", show_dataframe=False):
    """
    Plots a heatmap of the correlation matrix for numerical features.

    Args:
        df (pd.DataFrame): The dataset.
        selected_columns (list, optional): List of specific numerical columns to include in the heatmap.
        method (str): Correlation method - "pearson", "spearman", or "kendall". Default is "pearson".
        figsize (tuple): Size of the heatmap figure. Default is (10, 8).
        annot (bool): Whether to display correlation values inside the heatmap. Default is True.
        cmap (str): Colormap for the heatmap. Default is "coolwarm".
    """

    # Select numerical columns
    num_df = df.select_dtypes(include=["int64", "float64"])

    # If selected_columns are provided, filter them
    if selected_columns:
        num_df = num_df[selected_columns]

    if num_df.shape[1] < 2:
        print("⚠️ Not enough numerical features to compute correlation.")
        return

    # Compute correlation matrix
    corr_matrix = num_df.corr(method=method)

    # Option to return DataFrame instead of plotting
    if show_dataframe:
        return corr_matrix

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f"{method.capitalize()} Correlation Heatmap")
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_numeric_distributions(df, selected_columns=None, exclude_columns=None, hue_feature=None, bins=30, layout="multiple"):
    """
    Plots distributions for numerical features with optional hue feature grouping.

    Args:
        df (pd.DataFrame): The dataset.
        selected_columns (list, optional): List of specific numerical columns to plot.
        exclude_columns (list, optional): List of numerical columns to exclude from plotting.
        hue_feature (str, optional): Categorical feature for grouping (e.g., "Survived").
        bins (int): Number of bins for histograms.
        layout (str): "multiple" (default) to display one-by-one or "single" to show all in one figure.
    """

    num_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)  # Convert to list to avoid Index issues

    # Apply exclusions first
    if exclude_columns:
        num_cols = [col for col in num_cols if col not in exclude_columns]

    # If specific columns are requested, filter them
    if selected_columns:
        num_cols = [col for col in selected_columns if col in num_cols]

    # ✅ FIX: Use len(num_cols) instead of `if not num_cols`
    if len(num_cols) == 0:
        print("⚠️ No numerical columns found to plot!")
        return

    df_copy = df.copy()  # ✅ Ensure we don’t modify the original DataFrame

    if layout == "single":
        num_cols_count = len(num_cols)
        rows = math.ceil(num_cols_count / 2)

        fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            ax = axes[i]
            sns.histplot(data=df_copy, x=col, hue=hue_feature, bins=bins, kde=True, 
                         palette="coolwarm" if hue_feature else None, ax=ax, alpha=0.6)
            ax.set_title(f"Distribution of {col}" + (f" by {hue_feature}" if hue_feature else ""))
            
        if num_cols_count % 2 != 0:
            fig.delaxes(axes[-1])  # Remove empty subplot if an odd number of variables

        plt.tight_layout()
        plt.show()

    else:
        for col in num_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(data=df_copy, x=col, hue=hue_feature, bins=bins, kde=True, 
                         palette="coolwarm" if hue_feature else None, alpha=0.6)
            plt.title(f"Distribution of {col}" + (f" by {hue_feature}" if hue_feature else ""))
            plt.show()


if __name__ == "__main__":
    df = load_data()
    summarize_data(df)
    plot_distributions(df, exclude_columns=["passengerid", "survived"], layout="single")
    plot_categorical_distributions(df, layout="single", top_n=100)
    plot_categorical_distributions(df, selected_columns=["Pclass", "Sex", "Embarked"], hue_feature="Survived")
    plot_correlation_heatmap(df, method="kendall", cmap="viridis")
    plot_correlation_heatmap(df, selected_columns=["Age", "SibSp", "Parch", "Fare"])
    print("\nCorrelation:")
    corr_df = plot_correlation_heatmap(df, show_dataframe=True)
    print(corr_df)
    plot_numeric_distributions(df, layout="single", hue_feature="Survived")
    plot_numeric_distributions(df, layout="single", selected_columns=["Age", "Fare"], hue_feature="Survived", bins=10)


    

