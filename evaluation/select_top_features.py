import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def select_top_features(X, y, model, n_features=20):
    """
    Select top features based on model importance.
    
    Parameters:
        X: Input features.
        y: Target variable.
        model: Machine learning model.
        n_features: Number of top features to select (default=20).
        
    Returns:
        selected_features: List of selected feature indices.
    """
    # Fit the model to the data
    model.fit(X, y)

    # Extract feature importances from the model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    
    # Determine feature names based on input type
    num_features = X.shape[1]

    # Select top features based on importances
    top_feature_indices = importances.argsort()[::-1][:n_features]
    selected_features = [i for i in top_feature_indices]
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances, align='center')
    plt.xticks(range(len(importances)), range(num_features), rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()

    return selected_features
