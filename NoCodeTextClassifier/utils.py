from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def load_model(model_name):
    with open(os.path.join('models',model_name), 'rb') as f:
        model = pickle.load(f)
    return model


def prediction(model, X_test):
    model = load_model(model)
    y_pred = model.predict(X_test)
    return y_pred


def evaluation(model, X_test, y_test):
    y_pred = prediction(model, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    model_name = model.split(".")[0]
    print(f"Accuracy of {model_name}: {accuracy}\n")
    print(f"Classification Report of {model_name} : \n{class_report}\n")
    print(f"Confusion Matrix of {model_name} : \n{conf_matrix}")



