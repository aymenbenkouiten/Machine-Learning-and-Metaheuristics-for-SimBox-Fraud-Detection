import streamlit as st
import time
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
import sys
import os
sys.path.append(os.path.abspath(os.path.join('../')))
from evaluation.evaluate_lr_under import evaluate_lr
from evaluation.evaluate_dt_under import evaluate_dt
from evaluation.evaluate_rf_under import evaluate_rf
from evaluation.evaluate_xg_under import evaluate_xg
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Training using under sampling", page_icon="📉", layout='centered')
st.title("Training using under sampling")
st.write()

st.write("## Data spliting")

df = pd.read_csv('../datasets/final_features_dataset.csv')
X = df.drop(columns=['is_fraud', 'Unnamed: 0'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
y_train = y_train.values if isinstance(y_train, pd.Series) else y_train

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

train_index, val_index = next(cv.split(X_train, y_train))
X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
sampling_meth = RandomUnderSampler(random_state=0)
X_train_fold, y_train_fold = sampling_meth.fit_resample(X_train_fold, y_train_fold)

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Set of each fold")
    st.write("Training 60%")
    st.write("Validation 20%")
    st.write("Test 20%")

with col2:
    st.write("Fraudulent (1)")
    st.write(Counter(y_train_fold)[1] if 1 in Counter(y_train_fold) else 0)
    st.write(Counter(y_val_fold)[1] if 1 in Counter(y_val_fold) else 0)
    st.write(Counter(y_test)[1] if 1 in Counter(y_test) else 0)
with col3:
    st.write("Non-Fraudulent (0)")
    st.write(Counter(y_train_fold)[0] if 0 in Counter(y_train_fold) else 0)
    st.write(Counter(y_val_fold)[0] if 0 in Counter(y_val_fold) else 0)
    st.write(Counter(y_test)[0] if 0 in Counter(y_test) else 0)

def update_sliders(train_size, val_size, test_size):
    total = train_size + val_size + test_size
    if total != 100:
        # Adjust the sliders to maintain the total of 100%
        if train_size + val_size <= 99:
            test_size = 100 - (train_size + val_size)
        elif train_size + test_size <= 99:
            val_size = 100 - (train_size + test_size)
        else:
            train_size = 100 - (val_size + test_size)
    
    return train_size, val_size, test_size

st.write("## Algorithms")

with st.expander("Logistic Regression"):
    use_hho = st.checkbox("Use Harris Hawks Optimization to tune hyperparameters (HHO)")
    if use_hho:
        population_size = st.number_input("Population Size", min_value=1, step=1)
        iterations = st.number_input("Iterations", min_value=1, step=1)
    else:
        population_size = None
        iterations = None

    train_size = st.slider('Training Set Size (%)', 1, 99, 60)
    val_size = st.slider('Validation Set Size (%)', 1, 99, 20)
    test_size = st.slider('Test Set Size (%)', 1, 99, 20)
    train_size, val_size, test_size = update_sliders(train_size, val_size, test_size)

    # Display updated sliders
    st.write(f"Training Set Size (%): {train_size}")
    st.write(f"Validation Set Size (%): {val_size}")
    st.write(f"Test Set Size (%): {test_size}")
    if train_size>0 and val_size>0 and test_size>0:
        train_button_lr = st.button("Train")
    else:
        train_button_lr = st.button("Train", disabled=True)
    if train_button_lr:
        with st.spinner("Training Logistic Regression..."):
            evaluate_lr(use_hho=use_hho, population_size=population_size, iterations=iterations,
                        train_size=train_size, val_size=val_size, test_size=test_size)

with st.expander("Desicion Tree"):
    use_hho_dt = st.checkbox("Use Harris Hawks Optimization to tune hyperparameters (HHO) for decision tree")
    if use_hho_dt:
        population_size_dt = st.number_input("Population Size dt", min_value=1, step=1)
        iterations_dt = st.number_input("Iterations dt", min_value=1, step=1)
    else:
        population_size_dt = None
        iterations_dt = None

    train_size_dt = st.slider('Training Set Size (%) dt', 1, 99, 60)
    val_size_dt = st.slider('Validation Set Size (%) dt', 1, 99, 20)
    test_size_dt = st.slider('Test Set Size (%) dt', 1, 99, 20)
    train_size_dt, val_size_dt, test_size_dt = update_sliders(train_size_dt, val_size_dt, test_size_dt)

    # Display updated sliders
    st.write(f"Training Set Size (%): {train_size_dt}")
    st.write(f"Validation Set Size (%): {val_size_dt}")
    st.write(f"Test Set Size (%): {test_size_dt}")
    if train_size_dt>0 and val_size_dt>0 and test_size_dt>0:
        train_button_dt = st.button("Train  dt")
    else:
        train_button_dt = st.button("Train  dt", disabled=True)
    if train_button_dt:
        with st.spinner("Training Desicion Tree..."):
            evaluate_dt(use_hho=use_hho_dt, population_size=population_size_dt, iterations=iterations_dt,
                        train_size=train_size_dt, val_size=val_size_dt, test_size=test_size_dt)
            
with st.expander("Random Forest"):
    use_hho_rf = st.checkbox("Use Harris Hawks Optimization to tune hyperparameters (HHO) for Random Forest")
    if use_hho_rf:
        population_size_rf = st.number_input("Population Size rf", min_value=1, step=1)
        iterations_rf = st.number_input("Iterations rf", min_value=1, step=1)
    else:
        population_size_rf = None
        iterations_rf = None

    train_size_rf = st.slider('Training Set Size (%) rf', 1, 99, 60)
    val_size_rf = st.slider('Validation Set Size (%) rf', 1, 99, 20)
    test_size_rf = st.slider('Test Set Size (%) rf', 1, 99, 20)
    train_size_rf, val_size_rf, test_size_rf = update_sliders(train_size_rf, val_size_rf, test_size_rf)

    # Display updated sliders
    st.write(f"Training Set Size (%): {train_size_rf}")
    st.write(f"Validation Set Size (%): {val_size_rf}")
    st.write(f"Test Set Size (%): {test_size_rf}")
    if train_size_rf>0 and val_size_rf>0 and test_size_rf>0:
        train_button_rf = st.button("Train  rf")
    else:
        train_button_rf = st.button("Train  rf", disabled=True)
    if train_button_rf:
        with st.spinner("Training Random Forest..."):
            evaluate_rf(use_hho=use_hho_rf, population_size=population_size_rf, iterations=iterations_rf,
                        train_size=train_size_rf, val_size=val_size_rf, test_size=test_size_rf)
            
with st.expander("Xgboost"):
    use_hho_xg = st.checkbox("Use Harris Hawks Optimization to tune hyperparameters (HHO) for Xgboost")
    if use_hho_xg:
        population_size_xg = st.number_input("Population Size xg", min_value=1, step=1)
        iterations_xg = st.number_input("Iterations xg", min_value=1, step=1)
    else:
        population_size_xg = None
        iterations_xg = None

    train_size_xg = st.slider('Training Set Size (%) xg', 1, 99, 60)
    val_size_xg = st.slider('Validation Set Size (%) xg', 1, 99, 20)
    test_size_xg = st.slider('Test Set Size (%) xg', 1, 99, 20)
    train_size_xg, val_size_xg, test_size_xg = update_sliders(train_size_xg, val_size_xg, test_size_xg)

    # Display updated sliders
    st.write(f"Training Set Size (%): {train_size_xg}")
    st.write(f"Validation Set Size (%): {val_size_xg}")
    st.write(f"Test Set Size (%): {test_size_xg}")
    if train_size_xg>0 and val_size_xg>0 and test_size_xg>0:
        train_button_xg = st.button("Train  xg")
    else:
        train_button_xg = st.button("Train  xg", disabled=True)
    if train_button_xg:
        with st.spinner("Training XGboost..."):
            evaluate_xg(use_hho=use_hho_xg, population_size=population_size_xg, iterations=iterations_xg,
                        train_size=train_size_xg, val_size=val_size_xg, test_size=test_size_xg)