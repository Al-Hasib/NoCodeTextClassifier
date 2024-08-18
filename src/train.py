import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from preprocessing import *
from sklearn.svm import LinearSVC
import joblib

# Load train file
train_path = Path("./ML Engineer/train.csv")
df = pd.read_csv(train_path)

# convert the class attributes into number
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['class'])

# clean the training text
currency_symbols = r'[\$\£\€\¥\₹\¢\₽\₩\₪]'  
text_cleaner = TextCleaner(currency_symbols)
df['clean_text'] = df['email'].apply(lambda x: text_cleaner.clean_text(x))
print(df.head())

# TfIdfVectize the train data
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['target']
print(X.shape, y.shape)

# Initialize the classifier
svm_classifier = LinearSVC(C= 1,max_iter=1000, tol=0.0001)

# Train the model
svm_classifier.fit(X, y)

# Save the model to a file
joblib.dump(svm_classifier, 'email_detection_model.pkl')

print("Training Completed")