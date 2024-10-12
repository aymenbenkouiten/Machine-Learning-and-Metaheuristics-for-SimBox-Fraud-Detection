
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_feature_correlation(X, feature_names):
    """
    Calculate and visualize feature correlation using a heatmap.
    
    Parameters:
        X: Input features (DataFrame or ndarray).
        feature_names: Names of features (optional, for DataFrame input).
    """
    # Check input data type and calculate correlation matrix
    if isinstance(X, pd.DataFrame):
        correlation_matrix = X.corr()
    elif isinstance(X, np.ndarray):
        correlation_matrix = np.corrcoef(X, rowvar=False)
    else:
        raise ValueError("Input data type not supported. Please provide DataFrame or ndarray.")

    # Plot heatmap of feature correlation
    plt.figure(figsize=(10, 8))
    if feature_names is None:
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.xticks(ticks=np.arange(X.shape[1]) + 0.5, labels=np.arange(X.shape[1]) + 1, rotation=45)
        plt.yticks(ticks=np.arange(X.shape[1]) + 0.5, labels=np.arange(X.shape[1]) + 1, rotation=0)
    else:
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
                    xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Heatmap')
    plt.show()