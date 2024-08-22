# No Code Text Classifier Tool

This tool will help you to perform training, evaluation & prediction of Text Classification task without knowing any kind of code. You have to define the dataset directory and create your model and perform predictions without any issue. In the backend, this will automatically perform text preprocessing, model training etc. You can also perform hyperparameter techniques to get the best model through experiments. Let's get started.

Install the pakage
```python
pip install NoCodeTextClassifier
```

### Training the Text Classification

Define the datapath
```python
data_path = "dataset.csv"
```
Clean the Text dataset and transform the label into number
```python
# It will take datapath, text feature and target feature
process = process(data_path,'email','class')
df = process.processing()
print(df.head())
```
Convert the text feature into numerical vector. You can apply multiple vectorization such as TfIdfVectorizer, CountVectorizer.
```python
Vectorization = Vectorization(df,'clean_text')
TfidfVectorizer = Vectorization.TfidfVectorizer(max_features= 10000)
print(TfidfVectorizer.toarray())
```
Split the dataset into training and testing
```python
X_train, X_test, y_train, y_test = process.split_data(TfidfVectorizer.toarray(), df['labeled_target'])
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```
Perform training with various models such as Naive Bayers, Decision Tree, Logistic Regression, and others. After training, you will see the evalution of the trained model.
```python
models = Models(X_train=X_train,X_test = X_test, y_train = y_train, y_test = y_test)
models.DecisionTree()
```

### For Inferencing with text data

For prediction of the text data with the trained model, try this.
```python
text = input("Enter your text:\n")
inference.prediction(text)
```


Author:

**Md Abdullah Al Hasib**

### **Thank YOU**