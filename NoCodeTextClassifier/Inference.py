from NoCodeTextClassifier import preprocessing
from NoCodeTextClassifier import utils


def prediction(text):
    TextCleaner = preprocessing.TextCleaner()
    clean_text = TextCleaner.clean_text(text)

    vectorize = preprocessing.Vectorization()
    vectorize_text = vectorize.TfidfVectorizer(eval=True, string=clean_text)

    prediction = utils.prediction("DecisionTreeClassifier.pkl",vectorize_text)

    encoder = utils.load_artifacts("artifacts","encoder.pkl")
    output = encoder.inverse_transform(prediction)[0]

    print(f"The prediction of given text : \t{output}")
    




