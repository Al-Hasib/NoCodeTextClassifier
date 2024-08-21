from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier

from src.utils import *

class Models:
    def __init__(self, X_train,X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        os.makedirs("models",exist_ok=True)
        
    
    def DecisionTree(self, **kwargs):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        save_path = os.path.join("models", 'DecisionTreeClassifier.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print("Training Completed")
        evaluation('DecisionTreeClassifier.pkl', self.X_test,self.y_test)
        print("Finished")


    def LinearSVC(self, **kwargs):
        from sklearn.svm import LinearSVC
        model = LinearSVC(**kwargs)
        model.fit(self.X_train, self.y_train)
        save_path = os.path.join("models", 'LinearSVC.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        evaluation('LinearSVC.pkl', self.X_test,self.y_test)
        print("Training Completed")

    
    def RandomForestClassifier(self, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        save_path = os.path.join("models", 'RandomForestClassifier.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        evaluation('RandomForestClassifier.pkl', self.X_test,self.y_test)
        print("Training Completed")

    
    def MultinomialNB(self, **kwargs):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(**kwargs)
        model.fit(self.X_train, self.y_train)
        save_path = os.path.join("models", 'MultinomialNB.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

        evaluation('MultinomialNB.pkl', self.X_test,self.y_test)
        print("Training Completed")


