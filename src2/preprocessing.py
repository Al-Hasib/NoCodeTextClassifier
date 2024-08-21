import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    

    # functions for removing punctuations
    def remove_punctuation(self,text):
        return text.translate(str.maketrans('', '', string.punctuation))
    

    # Functions for cleaning text
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(self.currency_symbols, 'currency', text)
        text = self.remove_punctuation(text)
        text = re.compile('<.*?>').sub('', text)
        text = text.replace('_', '')
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        text = ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())
        
        return text

