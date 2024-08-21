import joblib
from src2.preprocessing import *
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path

# Input the email
text = input("Enter the Email: \n")

# load train data
train_path = Path("./ML Engineer/train.csv")
df = pd.read_csv(train_path)

# clean the text
currency_symbols = r'[\$\£\€\¥\₹\¢\₽\₩\₪]'  
text_cleaner = TextCleaner(currency_symbols)
df['clean_text'] = df['email'].apply(lambda x: text_cleaner.clean_text(x))

# fit the TfIdfVecotrizer with train data
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit(df['clean_text'])

# clean the input email
clean_text = str(text_cleaner.clean_text(text))
print(f"\nThe clean text is : {clean_text}")

# vectorize the clean email
y = vectorizer.transform([clean_text])

# Load the model from the file
loaded_model = joblib.load('email_detection_model.pkl')

# perform prediction of mail
predictions = int(loaded_model.predict(y)[0])
predictions = "spam" if predictions==1 else "not_spam"

# print the prediction
print(f"\nThe prediction is : {predictions}")