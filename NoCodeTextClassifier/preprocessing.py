import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from NoCodeTextClassifier import utils
import numpy as np


nltk.download('stopwords')
nltk.download('wordnet')

class TextCleaner:
    '''Class for cleaning Text'''
    def __init__(self, currency_symbols = r'[\$\£\€\¥\₹\¢\₽\₩\₪]', stop_words=None, lemmatizer=None):
        self.currency_symbols = currency_symbols
        
        if stop_words is None:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = stop_words
        
        if lemmatizer is None:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = lemmatizer

    def remove_punctuation(self,text):

        return text.translate(str.maketrans('', '', string.punctuation))
    

    # Functions for cleaning text
    def clean_text(self, text):
        '''
        Clean the text by removing punctuations, html tag, underscore, 
        whitespaces, numbers, stopwords.
        Lemmatize the words in root format.
        '''
        text = text.lower()
        text = re.sub(self.currency_symbols, 'currency', text)
        '''remove any kind of emojis in the text'''
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = self.remove_punctuation(text)
        text = re.compile('<.*?>').sub('', text)
        text = text.replace('_', '')
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        text = ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())
        #text = self.spell_correction(text)
        
        return str(text)


class process:
    def __init__(self, data_path:str, text_feature:str,target_feature:str):
        self.data_path = Path(data_path)
        self.text_feature = text_feature
        self.target_feature = target_feature

    def _read_data(self):
        df = pd.read_csv(self.data_path)
        return df
    
    def encoder_class(self, df):
        encoder = LabelEncoder()
        target = encoder.fit_transform(df[self.target_feature])
        os.makedirs("artifacts",exist_ok=True)
        save_path = os.path.join("artifacts", 'encoder.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(encoder, f)

        return target
    
    def clean_text(self, df):
        text_cleaner = TextCleaner()
        return df[self.text_feature].apply(lambda x: text_cleaner.clean_text(x))
    
    def processing(self):
        df = self._read_data()
        df['labeled_target'] = self.encoder_class(df)
        print("started Cleaning")
        df['clean_text'] = self.clean_text(df)
        return df
    
    def split_data(self, X, y):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    

class Vectorization:
    def __init__(self, dataframe=np.zeros((5,5)), text_feature='text_feature'):
        self.df = dataframe
        self.text = text_feature

        # Define the directory where you want to save the vectorizer
        self.vectorizer_dir = "vectorizers"



    def TfidfVectorizer(self, eval=False, string="text", **kwargs):
        # Step 1: Fit the Vectorizer on the Training Data
        vectorizer = TfidfVectorizer(**kwargs)
        if eval==True:
            tfidf_vectorizer = utils.load_artifacts("vectorizers","tfidf_vectorizer.pkl")
            return tfidf_vectorizer.transform([string])
            

        tfidf_vectorizer = vectorizer.fit_transform(self.df[self.text])
        print(tfidf_vectorizer.toarray().shape)
        os.makedirs(self.vectorizer_dir,exist_ok=True)
        save_path = os.path.join(self.vectorizer_dir, 'tfidf_vectorizer.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return tfidf_vectorizer
    
    def CountVectorizer(self, eval=False, string="text",**kwargs):
        # Step 1: Fit the Vectorizer on the Training Data
        vectorizer = CountVectorizer(**kwargs)
        if eval==True:
            tfidf_vectorizer = utils.load_artifacts("vectorizers","count_vectorizer.pkl")
            return tfidf_vectorizer.transform([string])
        count_vectorizer = vectorizer.fit_transform(self.df[self.text])
        print(count_vectorizer.toarray().shape)
        os.makedirs(self.vectorizer_dir,exist_ok=True)
        save_path = os.path.join(self.vectorizer_dir, 'count_vectorizer.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return count_vectorizer
    
    
    