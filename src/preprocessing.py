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
from sklearn.feature_extraction.text import TfidfVectorizer
from from_root import from_root


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
        return encoder.fit_transform(df[self.target_feature])
    
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
    def __init__(self, dataframe, text_feature):
        self.df = dataframe
        self.text = text_feature

        # Define the directory where you want to save the vectorizer
        self.vectorizer_dir = "vectorizers"



    def TfidfVectorizer(self, **kwargs):
        # Step 1: Fit the Vectorizer on the Training Data
        vectorizer = TfidfVectorizer(**kwargs)
        tfidf_vectorizer = vectorizer.fit_transform(self.df[self.text])
        print(tfidf_vectorizer.toarray().shape)
        os.makedirs(self.vectorizer_dir,exist_ok=True)
        save_path = os.path.join(self.vectorizer_dir, 'tfidf_vectorizer.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        return tfidf_vectorizer

        



if __name__=="__main__":
    data_path = r"C:\Users\abdullah\projects\NLP_project\NoCodeTextClassifier\ML Engineer\train.csv"

    process = process(data_path,'email','class')

    df = process.processing()

    print(df.head())

    Vectorization = Vectorization(df,'clean_text')

    TfidfVectorizer = Vectorization.TfidfVectorizer(max_features= 10000)

        # Step 3: Load the Saved Vectorizer
    with open(os.path.join('vectorizers','tfidf_vectorizer.pkl'), 'rb') as f:
        loaded_vectorizer = pickle.load(f)

    # Step 4: Transform the Test Data with the Loaded Vectorizer
    X_test_tfidf = loaded_vectorizer.transform(df['clean_text'])

    X_train, X_test, y_train, y_test = process.split_data(df['clean_text'], df['labeled_target'])

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Optional: Print the transformed test data
    print(X_test_tfidf.toarray().shape)