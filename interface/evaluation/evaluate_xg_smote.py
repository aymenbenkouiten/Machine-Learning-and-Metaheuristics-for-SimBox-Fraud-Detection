import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
from evaluation.hho import HarrisHawksOptimization
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import learning_curve
from joblib import dump
from xgboost import XGBClassifier
import streamlit as st
 
# Defining the hyperparameter search space
hyperparameter_space = {
    "n_estimators": [50, 500],
    "max_depth": [1, 30],
    "learning_rate": [0.01, 0.3],
    "subsample": [0.5, 1.0],
    "colsample_bytree": [0.3, 1.0],
    "min_child_weight": [1, 10],
    "gamma": [0, 0.5],
    "reg_alpha": [0, 1],
    "reg_lambda": [0, 1],
}
global_train_size = 0
global_val_size = 0
global_test_size = 0

def evaluate_xg(use_hho, population_size, iterations, train_size, val_size, test_size):
    starting = time.time()

    df = pd.read_csv('../datasets/final_features_dataset.csv')

    X = df.drop(columns=['is_fraud', 'Unnamed: 0'])
    y = df['is_fraud']

    global X_train, X_test, y_train, y_test, global_train_size, global_val_size, global_test_size
    global_train_size = train_size
    global_val_size = val_size
    global_test_size = test_size

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=0, stratify=y)
    X_train = X_train.values if isinstance(X, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y, pd.Series) else y_train

    if use_hho:
        hho = HarrisHawksOptimization(
            fitness,
            dimensions=len(hyperparameter_space),
            pop_size=population_size,
            lb=[hyperparameter_space[param][0] for param in hyperparameter_space],
            ub=[hyperparameter_space[param][1] for param in hyperparameter_space],
            max_iter=iterations
        )
        hho.optimize()
        plt.plot(range(1, len(hho.convergence_curve) + 1), hho.convergence_curve)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title('Convergence Curve')
        plt.grid(True)
        st.pyplot()

        model = fitness(hho.best_solution, True)
    else:
        model = XGBClassifier(random_state=0)
        model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=round((global_train_size+global_val_size)/global_val_size), shuffle=True, random_state=0)
    cv_results = []
    all_true = []
    all_pred = []
    all_proba = []
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train

    for train_index, val_index in cv.split(X_train, y_train):
        # Split data into training and testing sets
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        sampling_meth = SMOTE(random_state=0)
        X_train_fold, y_train_fold = sampling_meth.fit_resample(X_train_fold, y_train_fold)
        # Fit the model to the training data
        model = model.fit(X_train_fold, y_train_fold)
        
        # Predictions on training, testing, and validation sets
        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        
        # Calculate evaluation metrics
        f1_train = f1_score(y_train_fold, y_train_pred)
        f1_val = f1_score(y_val_fold, y_val_pred)
        f1_test = f1_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        accuracy = accuracy_score(y_test, y_test_pred)
        # Store evaluation results for this fold
        cv_results.append({'f1_train': f1_train, 'f1_test': f1_test, 'f1_val': f1_val,
                        'precision': precision, 'recall': recall, 'accuracy': accuracy,
                        'y_val_fold': y_val_fold, 'y_test_pred': y_test_pred, 'y_val_pred': y_val_pred})
        
        # Add true labels and predictions to overall lists
        all_true.extend(y_val_fold)
        all_pred.extend(y_val_pred)
        all_proba.extend(y_test_proba)
    # Calculate mean and standard deviation of evaluation metrics across folds§
    f1_trains = [fold['f1_train'] for fold in cv_results]
    f1_tests = [fold['f1_test'] for fold in cv_results]
    f1_vals = [fold['f1_val'] for fold in cv_results]
    precisions = [fold['precision'] for fold in cv_results]
    recalls = [fold['recall'] for fold in cv_results]
    accuracies = [fold['accuracy'] for fold in cv_results]
    for i, fold in enumerate(cv_results):
        st.write("Fold %d: (Train F1: %0.2f, Test F1: %0.2f, Validation F1: %0.2f, Precision: %0.2f, Recall: %0.2f, Accuracy: %0.2f)." 
            % (i+1, fold['f1_train'], fold['f1_test'], fold['f1_val'], fold['precision'], fold['recall'], fold['accuracy']))
        
    cm = confusion_matrix(all_true, all_pred)
    st.write(cm)
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
    st.pyplot()
    N, train_score, test_score = learning_curve(model, X, y, cv=cv, scoring='f1', train_sizes=np.linspace(0.1, 1, 10))
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, test_score.mean(axis=1), label='test score')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    st.pyplot()
    y_train_probabilities = model.predict_proba(X_train)[:, 1]
    y_test_probabilities = model.predict_proba(X_test)[:, 1]
    # Compute ROC curve and AUC for training set
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_probabilities)
    roc_auc_train = auc(fpr_train, tpr_train)
    # Compute ROC curve and AUC for testing set
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_probabilities)
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
    # Display the plot in Streamlit
    st.pyplot()

    model.fit(X, y)
    if use_hho:
        dump(model, "models/hho_xg_smote_f1_"+str(f1_tests_array.mean())+".joblib")
    else:
        dump(model, "models/xg_smote_f1_"+str(f1_tests_array.mean())+".joblib")
    
    exec_time = time.time() - starting

    st.write(exec_time, "seconds")


def fitness(hyperparameters, get_model = False):
    """
    Function to evaluate the fitness of decision tree model with given hyperparameters.

    Args:
    hyperparameters (list): List of hyperparameter values.
    get_model (bool): If True, returns the model only, otherwise returns the fitness score.

    Returns:
    float or DecisionTreeClassifier: Fitness score if get_model=False, else the model itself.
    """
    # Check if hyperparameters are within defined ranges
    for i, value in enumerate(hyperparameters):
        param_name = list(hyperparameter_space.keys())[i]
        range_min = hyperparameter_space[param_name][0]
        range_max = hyperparameter_space[param_name][1]
        if not (range_min <= value <= range_max):
            return float('-inf') # Penalize if hyperparameters are out of range

    # Create XGBoost model with specified hyperparameters
    model = XGBClassifier(
        n_estimators=int(hyperparameters[0]),
        max_depth=int(hyperparameters[1]),
        learning_rate=float(hyperparameters[2]),
        subsample=float(hyperparameters[3]),
        colsample_bytree=float(hyperparameters[4]),
        min_child_weight=float(hyperparameters[5]),
        gamma=float(hyperparameters[6]),
        reg_alpha=float(hyperparameters[7]),
        reg_lambda=float(hyperparameters[8]),
        random_state=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # If only model is requested, return the model
    if get_model:
        st.write('Best parameters', {
            "n_estimators": int(hyperparameters[0]),
            "max_depth": int(hyperparameters[1]),
            "learning_rate": float(hyperparameters[2]),
            "subsample": float(hyperparameters[3]),
            "colsample_bytree": float(hyperparameters[4]),
            "min_child_weight": float(hyperparameters[5]),
            "gamma": float(hyperparameters[6]),
            "reg_alpha": float(hyperparameters[7]),
            "reg_lambda": float(hyperparameters[8]),
            "use_label_encoder": False,
            "eval_metric": 'logloss'
        })
        return model
    else:
        # Initialize cross-validation
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        cv_results = []
        
        # Perform cross-validation
        for train_index, test_index in cv.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            # smote oversampling
            sm = SMOTE()
            X_train_fold, y_train_fold = sm.fit_resample(X_train_fold, y_train_fold)
            
            # Fit the model
            model = model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            y_train_pred = model.predict(X_train_fold)
            y_test_pred = model.predict(X_test_fold)
            
            # Calculate evaluation metrics
            f1_train = f1_score(y_train_fold, y_train_pred)
            f1_test = f1_score(y_test_fold, y_test_pred)

            # Store results
            cv_results.append({'f1_train': f1_train, 'f1_test': f1_test})
        
        # Extracting F1 scores for training, testing, and validation sets from cross-validation results
        f1_train = np.array([fold["f1_train"] for fold in cv_results])
        f1_test = np.array([fold["f1_test"] for fold in cv_results])
        
        # Initialize the score variable to negative infinity
        score = float("-inf")
        # Check if the standard deviation of the mean F1 scores across training, testing, and validation sets is less than 0.01
        if np.array([f1_train.mean(), f1_test.mean()]).std()<0.001:
            # If the condition is met, set the score to the mean F1 score of the validation set
            score = f1_test.mean()
        return score