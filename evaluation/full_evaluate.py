from sklearn.model_selection import train_test_split, StratifiedKFold
import os
from evaluation.find_best_params import find_best_params
from evaluation.cross_validation import cross_validation
from evaluation.plot_learning_curve import plot_learning_curve
from evaluation.plot_ROC_AUC import plot_ROC_AUC
from evaluation.calculate_feature_correlation import calculate_feature_correlation
from evaluation.select_top_features import select_top_features
from joblib import dump

def full_evaluate(eval, X, y, hyperparameter_space, sampling=None, max_iter=10, pop_size=5):
    """
    Full evaluation of a machine learning model.
    
    Parameters:
        eval: Evaluation method/function for the model.
        X: Input features.
        y: Target variable.
        hyperparameter_space: Dictionary of hyperparameters to search through.
        sampling: Optional parameter to specify data sampling technique (default=None).
        max_iter: Maximum number of iterations for hyperparameter search (default=10).
        pop_size: Population size for hyperparameter search (default=5).
    """
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    # Find the best hyperparameters for the model
    best_solution = find_best_params(eval, hyperparameter_space, max_iter, pop_size).best_solution

    # Train the model with the best hyperparameters
    model = eval(best_solution, True)
    
    # Define filename for saving the trained model
    if sampling is not None:
        filename = '../../models/'+model.__class__.__name__ + '_' + sampling + '.joblib'
    else:
        filename = '../../models/'+model.__class__.__name__ + '.joblib'

    # Setup cross-validation for the model with best hyperparameters
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    # Perform cross-validation 
    cross_validation(model, X_train, X_val, y_train, y_val, cv, sampling=sampling)
    # Plot learning curve
    plot_learning_curve(model, X_train, y_train, cv)
    # Plot ROC-AUC curve
    plot_ROC_AUC(model, X_train, X_val, y_train, y_val)

    # Train then model on the whole dataset then save the trained model
    model.fit(X, y)
    dump(model, filename)
    
