
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_ROC_AUC(model, X_train, X_val, y_train, y_val):
    """
    Plot ROC curves and calculate AUC for training and testing sets.
    
    Parameters:
        model: Machine learning model.
        X_train: Training features.
        X_val: Validation features.
        y_train: Training labels.
        y_val: Validation labels.
    """
    # Calculate probabilities for positive class
    y_train_probabilities = model.predict_proba(X_train)[:, 1]
    y_test_probabilities = model.predict_proba(X_val)[:, 1]

    # Compute ROC curve and AUC for training set
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_probabilities)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Compute ROC curve and AUC for testing set
    fpr_test, tpr_test, thresholds_test = roc_curve(y_val, y_test_probabilities)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curves
    plt.figure(figsize=(12, 6))

    # Plot ROC curve for training set
    plt.subplot(1, 2, 1)
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'Training ROC curve (AUC = {roc_auc_train:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    # Plot ROC curve for testing set
    plt.subplot(1, 2, 2)
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Testing ROC curve (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    
    # Adjust layout and display plots
    plt.tight_layout()
    plt.show()