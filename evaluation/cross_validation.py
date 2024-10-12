from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from evaluation.plot_confusion_matrix import plot_confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def cross_validation(model, X_train, X_val, y_train, y_val, cv, sampling):
    """
    Perform cross-validation to evaluate the performance of a machine learning model.
    
    Parameters:
        model: Machine learning model to evaluate.
        X_train: Training features.
        X_val: Validation features.
        y_train: Training labels.
        y_val: Validation labels.
        cv: Cross-validation iterator.
    """
    # Initialize lists to store evaluation results
    cv_results = []
    all_true = []
    all_pred = []
    all_proba = []

    # Convert data to arrays if they are not already
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train

    # Loop through each fold of cross-validation
    for train_index, test_index in cv.split(X_train, y_train):
        # Split data into training and testing sets
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # Apply data sampling technique if specified
        if sampling == "smote":
            sampling_meth = SMOTE(random_state=0)
            X_train_fold, y_train_fold = sampling_meth.fit_resample(X_train_fold, y_train_fold)
        elif sampling == "under":
            sampling_meth = RandomUnderSampler(random_state=0)
            X_train_fold, y_train_fold = sampling_meth.fit_resample(X_train_fold, y_train_fold)

        # Print class distribution in each set
        print("Training set", Counter(y_train_fold), "Test set", Counter(y_test_fold), "Validation set", Counter(y_val))

        # Fit the model to the training data
        model = model.fit(X_train_fold, y_train_fold)
        
        # Predictions on training, testing, and validation sets
        y_train_pred = model.predict(X_train_fold)
        y_test_pred = model.predict(X_test_fold)
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)
        
        # Calculate evaluation metrics
        f1_train = f1_score(y_train_fold, y_train_pred)
        f1_test = f1_score(y_test_fold, y_test_pred)
        f1_val = f1_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        accuracy = accuracy_score(y_val, y_val_pred)

        # Store evaluation results for this fold
        cv_results.append({'f1_train': f1_train, 'f1_test': f1_test, 'f1_val': f1_val,
                           'precision': precision, 'recall': recall, 'accuracy': accuracy,
                           'y_test_fold': y_test_fold, 'y_test_pred': y_test_pred, 'y_val_pred': y_val_pred})
        
        # Add true labels and predictions to overall lists
        all_true.extend(y_test_fold)
        all_pred.extend(y_test_pred)
        all_proba.extend(y_val_proba)

    # Calculate mean and standard deviation of evaluation metrics across folds
    f1_trains = [fold['f1_train'] for fold in cv_results]
    f1_tests = [fold['f1_test'] for fold in cv_results]
    f1_vals = [fold['f1_val'] for fold in cv_results]
    precisions = [fold['precision'] for fold in cv_results]
    recalls = [fold['recall'] for fold in cv_results]
    accuracies = [fold['accuracy'] for fold in cv_results]

    # Print evaluation metrics for each fold
    for i, fold in enumerate(cv_results):
        print("Fold %d: (Train F1: %0.2f, Test F1: %0.2f, Validation F1: %0.2f, Precision: %0.2f, Recall: %0.2f, Accuracy: %0.2f)." 
              % (i+1, fold['f1_train'], fold['f1_test'], fold['f1_val'], fold['precision'], fold['recall'], fold['accuracy']))
    
    # Plot confusion matrix for overall performance
    cm = confusion_matrix(all_true, all_pred)
    plt.figure()
    plot_confusion_matrix(cm, classes=np.unique(y_train), title='Confusion matrix - Overall')

    # Plot bar chart of mean evaluation metrics across folds with error bars
    f1_trains_array = np.array(f1_trains)
    f1_tests_array = np.array(f1_tests)
    f1_vals_array = np.array(f1_vals)
    precisions_array = np.array(precisions)
    recalls_array = np.array(recalls)
    accuracies_array = np.array(accuracies)
    metrics = ['Train F1', 'Test F1', 'Validation F1', 'Precision', 'Recall', 'Accuracy']
    means = [f1_trains_array.mean(), f1_tests_array.mean(), f1_vals_array.mean(), 
             precisions_array.mean(), recalls_array.mean(), accuracies_array.mean()]
    stds = [f1_trains_array.std(), f1_tests_array.std(), f1_vals_array.std(), 
            precisions_array.std(), recalls_array.std(), accuracies_array.std()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.ylabel('Score')
    plt.title('Cross-Validation Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
