import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from NoCodeTextClassifier.EDA import Informations, Visualizations

st.title('No Code Text Classification App')
st.write('Understand the behavior of your text data and train a model to classify the text data')
section = st.sidebar.radio("Choose Section", ["Data Analysis", "Train Model", "Predictions"])
# CSV upload
# Upload Data
st.sidebar.subheader("Upload Your Dataset")
train_data = st.sidebar.file_uploader("Upload training data", type=["csv"])
test_data = st.sidebar.file_uploader("Upload test data", type=["csv"])

if train_data is not None and test_data is not None:
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)
    st.write("Training Data")
    st.write(train_df.head(3))
    columns = train_df.columns.tolist()
    text_data = st.sidebar.selectbox("Choose the text column:", columns)
    target = st.sidebar.selectbox("Choose the target column:", columns)



    

if section=="Data Analysis":
    st.subheader("Get Insights from the Data")
    info = Informations(train_df, text_data, target)
    st.write("Data Shape:", info.shape())
    st.write("Class Imbalance:", info.class_imbalanced())
    st.write("Missing Values:", info.missing_values())
    train_df['clean_text'] = info.clean_text()
    train_df['text_length'] = info.text_length()
    train_df['target'] = info.label_encoder()
    st.write(train_df.head(3))
    st.write(info.analysis_text_length('text_length'))
    st.write("Correlation between Text Length and Target:", info.correlation('text_length'))


    st.subheader("Visualizations")
    vis = Visualizations(train_df, text_data, target)
    vis.class_distribution()
    vis.text_length_distribution()

if section=="Train Model":
    st.subheader("Train a Model")
    st.radio("Choose ", ["Logistic Regression", "Random Forest"])
    

if section=="Predictions":
    st.subheader("Perform Predictions on the Test Data")
    
