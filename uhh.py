import streamlit as st
import pandas as pd
import plotly.express as px

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="UAC Care Load Forecast", layout="wide")

# FIRST load data
df = pd.read_csv("uac_data_processed.csv")

st.title("UAC Program - Predictive Capacity Dashboard")

# THEN create metrics
col1,col2,col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Max HHS Care", int(df["Children in HHS Care"].max()))
col3.metric("Avg Daily Discharge",
            round(df["Children discharged from HHS Care"].mean(),2))

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="UAC Care Load Forecast", layout="wide")

st.title("UAC Program - Predictive Capacity Dashboard")

df = pd.read_csv("uac_data_processed.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Care Load Trend
if "Children in HHS Care" in df.columns:
    fig = px.line(
        df,
        x="Date",
        y="Children in HHS Care",
        title="Children in HHS Care Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

# Discharge Trend
if "Children discharged from HHS Care" in df.columns:
    fig2 = px.line(
        df,
        x="Date",
        y="Children discharged from HHS Care",
        title="Children Discharged from HHS Care"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Model Evaluation")

results = pd.DataFrame({
    "Model":["Naive","Random Forest","Gradient Boosting","Exp. Smoothing","SARIMA"],
    "MAE":[6.62,33.96,48.26,309.46,480.15],
    "RMSE":[8.24,52.97,70.02,425.14,668.43],
    "MAPE":[0.29,1.61,2.29,13.32,20.79]
})

st.dataframe(results)

st.success("Best Performing Model: Naive Forecast")