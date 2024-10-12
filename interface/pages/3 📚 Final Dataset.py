import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(page_title="Final Dataset", page_icon="📚", layout='centered')
df = pd.read_csv('../datasets/final_features_dataset.csv')
df = df.rename(columns={'Unnamed: 0': 'Phone number'})

st.title("Final Dataset")
st.write()

counts = Counter(df["is_fraud"])

st.write("### Counts of Fraudulent and Non-Fraudulent")

col1, col2 = st.columns(2)
with col1:
    st.write("Fraudulent (1)")
    st.write(counts[1] if 1 in counts else 0)
with col2:
    st.write("Non-Fraudulent (0)")
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