
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(model, X_train, y_train, cv):
    """
    Plot learning curve.
    
    Parameters:
        model: Machine learning model.
        X_train: Training features.
        y_train: Training labels.
        cv: Cross-validation iterator.
    """
    # Calculate learning curve
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv=cv, scoring='f1', train_sizes=np.linspace(0.1, 1, 10))

    # Plot learning curve
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()