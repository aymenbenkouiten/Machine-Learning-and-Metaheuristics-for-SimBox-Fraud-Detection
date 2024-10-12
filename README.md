# Machine Learning and Metaheuristics for SimBox Fraud Detection

## Project Overview

This project focuses on applying **machine learning** and **metaheuristic optimization techniques** to enhance fraud detection systems in the telecommunications sector, particularly targeting **SIMBox fraud**. The project uses a variety of machine learning algorithms, such as Logistic Regression, Decision Trees, Random Forest, and XGBoost, optimized using the **Harris Hawks Optimization (HHO)** algorithm.

## Summary

In today’s digital age, telecommunications play a vital role in our daily lives. However, the growing dependence on these technologies exposes operators to sophisticated fraud, particularly **SIMBox fraud**, which causes financial losses and compromises customer trust. This project aims to optimize fraud detection by parameterizing machine learning models and using the **Harris Hawks Optimization (HHO)** metaheuristic to enhance performance. The final model demonstrated high accuracy, showing potential for real-world applications.

## Key Results

- Applied **Logistic Regression**, **Decision Trees**, **Random Forest**, and **XGBoost** to detect fraudulent SIMBox activity.
- Used **Harris Hawks Optimization** (HHO) to fine-tune model hyperparameters for optimal performance.
- The **XGBoost** model without resampling showed the highest performance, with an **F1-score** of 0.98 and **recall** of 0.95.

## Data

- The dataset includes telecommunications data with labeled instances of **fraudulent** and **non-fraudulent** activities.
- Data was split into **60% training**, **20% validation**, and **20% testing**.
- Resampling techniques, such as **SMOTE** and under-sampling, were used to address class imbalance.

## Tools and Technologies

- **Programming Languages**: Python
- **Libraries**: Scikit-learn, XGBoost, Pandas, Numpy, Matplotlib
- **Metaheuristic**: Harris Hawks Optimization (HHO)
- **Modeling Techniques**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Resampling**: SMOTE, under-sampling

## Project Structure

```plaintext
.
├── datasets
│   ├── PFE_DATASET_FRAUD.csv         # Dataset labeled as fraud
│   ├── PFE_DATASET_NOT_FRAUD.csv     # Dataset labeled as non-fraud
│   └── final_features_dataset.csv    # Final dataset with selected features
├── evaluation
│   └── [evaluation scripts]          # Custom scripts for model evaluation
├── GUI
│   └── [graphical interface]         # Interface for interacting with project results
├── models
│   └── [best models with HHO]        # Models optimized with HHO
├── train
│   ├── under-sampling                # Notebooks for training models with under-sampling
│   ├── smote-sampling                # Notebooks for training models with SMOTE
│   └── without-sampling              # Notebooks for training models without sampling
├── hho.py                            # Harris Hawks Optimization algorithm script
├── features-engineering.ipynb        # Feature selection and transformation notebook
├── exploratory-data-analysis.ipynb   # Data exploration and visualization notebook
└── deployment.ipynb                  # System deployment notebook

```

## Model Performance

Here are the final results obtained for each model after applying various resampling techniques:

```plaintext
|-------------------------|----------------|----------|----------------|---------|--------|
| Model                   | Sampling       | Train F1 | Validation F1  | Test F1 | Recall |
|-------------------------|----------------|----------|----------------|---------|--------|
| **Logistic Regression** | Without        | 0.08     | 0.08           | 0.07    | 0.45   |
|                         | With SMOTE     | 0.56     | 0.07           | 0.07    | 0.45   |
|                         | Under-sampling | 0.50     | 0.07           | 0.07    | 0.42   |
| **Decision Tree**       | Without        | 0.97     | 0.97           | 0.95    | 0.89   |
|                         | With SMOTE     | 0.92     | 0.92           | 0.90    | 0.83   |
|                         | Under-sampling | 0.92     | 0.92           | 0.90    | 0.83   |
| **Random Forest**       | Without        | 0.98     | 0.97           | 0.96    | 0.92   |
|                         | With SMOTE     | 0.92     | 0.92           | 0.91    | 0.83   |
|                         | Under-sampling | 0.93     | 0.93           | 0.92    | 0.84   |
| **XGBoost**             | Without        | 0.98     | 0.98           | 0.98    | 0.95   | **Best model** |
|                         | With SMOTE     | 0.93     | 0.93           | 0.91    | 0.84   |
|                         | Under-sampling | 0.93     | 0.93           | 0.91    | 0.84   |
|-------------------------|----------------|----------|----------------|---------|--------|
