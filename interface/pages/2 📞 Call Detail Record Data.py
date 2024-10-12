import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(page_title="Call Detail Record Data", page_icon="📞", layout='centered')

fraud_df = pd.read_csv('../datasets/PFE_DATASET_FRAUD.CSV', delimiter=';')
fraud_df["is_fraud"] = 1
not_fraud_df = pd.read_csv('../datasets/PFE_DATASET_NOT_FRAUD.CSV', delimiter=';')
not_fraud_df["is_fraud"] = 0
df = pd.concat([fraud_df, not_fraud_df])

st.title("Call Detail Record Data")
st.write()

counts = Counter(df.groupby("A_Num")["is_fraud"].any().astype(int).reset_index()["is_fraud"])

st.write("### Counts of Fraudulent and Non-Fraudulent")

col1, col2 = st.columns(2)
with col1:
    st.write("Fraudulent (1)")
    st.write(len(fraud_df))
    st.write(counts[1] if 1 in counts else 0)
with col2:
    st.write("Non-Fraudulent (0)")
    st.write(len(not_fraud_df))
    st.write(counts[0] if 0 in counts else 0)

st.write("### Call DataFrame")
st.dataframe(df.head(10))

if st.checkbox('Show DataFrame Description'):
    st.write(df.describe())

if st.checkbox('Show DataFrame Info'):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

if st.checkbox('Show Charging Time Distribution before the filter'):
    fraud_df['Charging_Tm'] = pd.to_datetime(fraud_df['Charging_Tm'])
    not_fraud_df['Charging_Tm'] = pd.to_datetime(not_fraud_df['Charging_Tm'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(fraud_df['Charging_Tm'], label='fraud_df', fill=True, kde=True, ax=ax)
    sns.histplot(not_fraud_df['Charging_Tm'], label='not_fraud_df', fill=True, kde=True, ax=ax)
    ax.set_title('Distribution of Charging Time')
    ax.legend()
    st.pyplot(fig)

if st.checkbox('Show Charging Time Distribution after the filter'):
    fraud_df['Charging_Tm'] = pd.to_datetime(fraud_df['Charging_Tm'])
    not_fraud_df['Charging_Tm'] = pd.to_datetime(not_fraud_df['Charging_Tm'])
    fraud_df = fraud_df[(fraud_df["Charging_Tm"] > pd.to_datetime("22/04/2024")) & (fraud_df["Charging_Tm"] < pd.to_datetime("25/04/2024"))]
    not_fraud_df = not_fraud_df[(not_fraud_df["Charging_Tm"] > pd.to_datetime("22/04/2024")) & (not_fraud_df["Charging_Tm"] < pd.to_datetime("25/04/2024"))]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(fraud_df['Charging_Tm'], label='fraud_df', fill=True, kde=True, ax=ax)
    sns.histplot(not_fraud_df['Charging_Tm'], label='not_fraud_df', fill=True, kde=True, ax=ax)
    ax.set_title('Distribution of Charging Time')
    ax.legend()
    st.pyplot(fig)
    counts = Counter(pd.concat([fraud_df, not_fraud_df]).groupby("A_Num")["is_fraud"].any().astype(int).reset_index()["is_fraud"])

    st.write("Counts of Fraudulent and Non-Fraudulent")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Fraudulent (1)")
        st.write(len(fraud_df))
        st.write(counts[1] if 1 in counts else 0)
    with col2:
        st.write("Non-Fraudulent (0)")
        st.write(len(not_fraud_df))
        st.write(counts[0] if 0 in counts else 0)