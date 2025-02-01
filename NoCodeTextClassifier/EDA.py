import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from NoCodeTextClassifier.preprocessing import TextCleaner
from sklearn.preprocessing import LabelEncoder


class Informations:
    def __init__(self, data, text_data, target):
        self.data = data
        self.text_data = text_data
        self.target = target
    
    def shape(self):
        return self.data.shape
    
    def class_imbalanced(self):
        return self.data[self.target].value_counts()
    
    def missing_values(self):
        return self.data.isnull().sum()
    
    def label_encoder(self):
        encoder = LabelEncoder()
        target = encoder.fit_transform(self.data[self.target])
        return target
    
    def clean_text(self):
        text_cleaner = TextCleaner()
        return self.data[self.text_data].apply(lambda x: text_cleaner.clean_text(x))

    def text_length(self):
        return self.data[self.text_data].apply(lambda x: len(x))
    
    def analysis_text_length(self, text_length):
        result = self.data[text_length].describe()
        return result
    
    def correlation(self, other_feature):
        return self.data[other_feature].corr(self.data["target"])
    
        
    
    

class Visualizations:
    def __init__(self, data, text_data, target):
        self.data = data
        self.text_data = text_data
        self.target = target
    
    def simple_plot(self):
        # Generate sample data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        fig, ax = plt.subplots()
        ax.plot(x, y, label="Sine Wave")
        ax.set_title("Matplotlib Plot in Streamlit")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()
        # Display the plot in Streamlit
        st.pyplot(fig)
    
    def class_distribution(self):
        fig, ax = plt.subplots()
        sns.countplot(x=self.data[self.target], ax=ax,palette="pastel")
        ax.set_title("Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    def text_length_distribution(self):
        fig, ax = plt.subplots()
        sns.histplot(self.data['text_length'], ax=ax, kde=True)
        ax.set_title("Text Length Distribution")
        ax.set_xlabel("Text Length")
        ax.set_ylabel("Count")
        st.pyplot(fig)


    

    

    