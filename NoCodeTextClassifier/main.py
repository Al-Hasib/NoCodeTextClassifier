from NoCodeTextClassifier.preprocessing import *
from NoCodeTextClassifier.models import *


if __name__=="__main__":
    data_path = r"C:\Users\abdullah\projects\NLP_project\NoCodeTextClassifier\ML Engineer\train.csv"

    process = process(data_path,'email','class')

    df = process.processing()

    print(df.head())

    Vectorization = Vectorization(df,'clean_text')

    TfidfVectorizer = Vectorization.TfidfVectorizer(max_features= 10000)
    print(TfidfVectorizer.toarray())
    X_train, X_test, y_train, y_test = process.split_data(TfidfVectorizer.toarray(), df['labeled_target'])

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # print(X_train, y_train)
    models = Models(X_train=X_train,X_test = X_test, y_train = y_train, y_test = y_test)

    models.DecisionTree()