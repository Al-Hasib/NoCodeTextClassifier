import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from NoCodeTextClassifier.EDA import Informations, Visualizations
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from NoCodeTextClassifier.preprocessing import process, TextCleaner, Vectorization  
from NoCodeTextClassifier.models import Models
import os
from NoCodeTextClassifier.utils import *

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

    info = Informations(train_df, text_data, target)
    train_df['clean_text'] = info.clean_text()
    train_df['text_length'] = info.text_length()
    train_df['target'] = info.label_encoder()

    # test_info = Informations(test_df, text_data, target)
    # test_df['clean_text'] = test_info.clean_text()
    # test_df['text_length'] = test_info.text_length()



    

if section=="Data Analysis":
    try:
        st.subheader("Get Insights from the Data")
        
        st.write("Data Shape:", info.shape())
        st.write("Class Imbalance:", info.class_imbalanced())
        st.write("Missing Values:", info.missing_values())

        st.write(train_df.head(3))
        st.markdown("**Text Length Analysis**")
        st.write(info.analysis_text_length('text_length'))
        st.write("Correlation between Text Length and Target:", info.correlation('text_length'))


        st.subheader("Visualizations")
        vis = Visualizations(train_df, text_data, target)
        vis.class_distribution()
        vis.text_length_distribution()

    except Exception as e:
        st.write("Please upload the data to get the insights")
        

if section=="Train Model":
    try:
        st.subheader("Train a Model")

        
        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            model = st.radio("Choose the Model", ["Logistic Regression","Decision Tree", 
                            "Random Forest", "Linear SVC", "SVC",
                            "Multinomial Naive Bayes", "Gaussian Naive Bayes"])
        with col2:
            vectorizer = st.radio("Choose Vectorizer", ["Tfidf Vectorizer", "Count Vectorizer"])

        if vectorizer=="Tfidf Vectorizer":
            vectorizer = TfidfVectorizer(max_features=10000)
        else:
            vectorizer = CountVectorizer(max_features=10000)
        
        st.write(train_df.head(3))
        X = vectorizer.fit_transform(train_df['clean_text'])
        y = train_df['target']
        X_train, X_test, y_train, y_test = process.split_data(X, y)
        st.write(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        if st.button("Start Training"):
            models = Models(X_train=X_train,X_test = X_test, y_train = y_train, y_test = y_test)
            
            if model=="Logistic Regression":
                models.LogisticRegression()
            elif model=="Decision Tree":
                models.DecisionTree()
            elif model=="Linear SVC":
                models.LinearSVC()
            elif model=="SVC":
                models.SVC()
            elif model=="Multinomial Naive Bayes":
                models.MultinomialNB()
            elif model=="Random Forest":
                models.RandomForestClassifier()
            else:
                models.GaussianNB()

    except Exception as e:
        st.write("Please upload the data to train the model")
    

if section=="Predictions":
    try:
        st.subheader("Perform Predictions on the Test Data")
        text = st.text_area("Enter the text to classify")

        select_model = os.listdir("models")
        model = st.selectbox("Choose the Model", select_model)
        if st.button("Predict"):
            TextCleaner = TextCleaner()
            clean_text = TextCleaner.clean_text(text)

            vectorize = Vectorization()
            vectorize_text = vectorize.TfidfVectorizer(eval=True, string=clean_text)

            prediction = prediction(model, vectorize_text)

            encoder = load_artifacts("artifacts","encoder.pkl")
            output = encoder.inverse_transform(prediction)[0]
            st.markdown(f"**Text**: {text}\n")
            st.markdown(f"Prediction: **{output}**")

    except Exception as e:
        print(e)
        st.write("Please Train the Model to perform the predictions")

