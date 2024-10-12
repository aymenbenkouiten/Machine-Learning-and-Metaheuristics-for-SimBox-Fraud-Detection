import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PowerTransformer
import os
import numpy as np

st.set_page_config(page_title="Prediction using CDR Dataset file", page_icon="📤", layout='centered')

st.title("Fraud Detection Prediction")

# Load models from the models folder
model_folder = "models"
model_files = [f for f in os.listdir(model_folder) if f.endswith(".joblib")]

if not model_files:
    st.error("No model files found in the 'models' folder.")
else:
    selected_model_file = st.selectbox("Select a model", model_files)
    model_path = os.path.join(model_folder, selected_model_file)
    model = joblib.load(model_path)

    delimiter = st.selectbox("Select the delimiter used in your CSV file", [",", ";", "\t", "|"])
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file, delimiter=delimiter)

            st.write("Uploaded Data:")
            st.write(df.head())

            df['Charging_Tm'] = pd.to_datetime(df['Charging_Tm'])


            features_df = pd.DataFrame()
            features_df[['AVG_CALL_FREQ', 'STD_CALL_FREQ']] = df.sort_values(['A_Num', 'Charging_Tm']).groupby('A_Num')['Charging_Tm'].diff().dt.total_seconds().groupby(df.sort_values(['A_Num', 'Charging_Tm'])['A_Num']).agg(['mean', 'std'])
            for des in ["ON-NET", "OFF-NET", "INTERNATIONAL"]:
                features_df[des.replace('-', '_')+'_RATIO'] = df.groupby('A_Num').agg(ON_NET_RATIO=('DESTINATION_CAT', lambda x: (x==des).sum() / len(x)))
            for call_type in [0, 1, 6, 7]:
                features_df['CALLTYPE_'+str(call_type)+'_RATIO'] = df.groupby('A_Num').agg(TELESRVC11_RATIO=('Call_Type', lambda x: (x==call_type).sum() / len(x)))
            features_df['MIN_CALL_FREQ'] = df.groupby('A_Num').apply(lambda x: x.sort_values('Charging_Tm')).reset_index(drop=True).assign(INTERVAL_ACTIVITIES_SECONDS=lambda x: x.groupby('A_Num')['Charging_Tm'].diff().dt.total_seconds()).groupby('A_Num')['INTERVAL_ACTIVITIES_SECONDS'].min()
            features_df['AVG_CALL_FREQ'] = df.groupby('A_Num').apply(lambda x: x.sort_values('Charging_Tm')).reset_index(drop=True).assign(INTERVAL_ACTIVITIES_SECONDS=lambda x: x.groupby('A_Num')['Charging_Tm'].diff().dt.total_seconds()).groupby('A_Num')['INTERVAL_ACTIVITIES_SECONDS'].mean()
            features_df['MAX_CALL_FREQ'] = df.groupby('A_Num').apply(lambda x: x.sort_values('Charging_Tm')).reset_index(drop=True).assign(INTERVAL_ACTIVITIES_SECONDS=lambda x: x.groupby('A_Num')['Charging_Tm'].diff().dt.total_seconds()).groupby('A_Num')['INTERVAL_ACTIVITIES_SECONDS'].max()
            features_df['MIN_DAILY_CALLS'] = df.groupby(['A_Num', df['Charging_Tm'].dt.date])['A_Num'].count().groupby('A_Num').min()
            features_df['AVG_DAILY_CALLS'] = df.groupby(['A_Num', df['Charging_Tm'].dt.date])['A_Num'].count().groupby('A_Num').mean()
            features_df['MAX_DAILY_CALLS'] = df.groupby(['A_Num', df['Charging_Tm'].dt.date])['A_Num'].count().groupby('A_Num').max()
            features_df['MIN_DAILY_CALLS_DUR'] = df.groupby(['A_Num', df['Charging_Tm'].dt.date])['Call_Duration'].sum().groupby('A_Num').min()
            features_df['AVG_DAILY_CALLS_DUR'] = df.groupby(['A_Num', df['Charging_Tm'].dt.date])['Call_Duration'].sum().groupby('A_Num').mean()
            features_df['MAX_DAILY_CALLS_DUR'] = df.groupby(['A_Num', df['Charging_Tm'].dt.date])['Call_Duration'].sum().groupby('A_Num').max()

            features_df['A_Num'] = features_df.index

            features_df = features_df.fillna(0)

            st.write("Features Data:")
            st.write(features_df.head())

            features = features_df.columns.difference(['A_Num', 'is_fraud'])

            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            transformed = pt.fit_transform(features_df.drop(columns=['A_Num']))
            transformed_df = pd.DataFrame(transformed, columns=features)

            
            st.write("Transformed Data:")
            st.write(transformed_df.head())

            a_num_df = features_df[["A_Num"]]
            predictions = model.predict(transformed_df)
            predictions_df = pd.DataFrame(predictions, columns=["Predictions"])
            result_df = pd.concat([a_num_df.reset_index(drop=True), predictions_df], axis=1)

            unique_a_nums = result_df["A_Num"]
            random_numbers = ["07" + str(np.random.randint(70000000, 99999999)) for _ in unique_a_nums]
            a_num_mapping = dict(zip(unique_a_nums, random_numbers))
            result_df["Phone number"] = result_df["A_Num"].map(a_num_mapping)

            st.write("Predictions:")
            st.write(result_df[["A_Num", "Phone number", "Predictions"]])
            
        except Exception as e:
            st.error(f"An error occurred: {e}")