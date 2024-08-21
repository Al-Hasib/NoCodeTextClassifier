from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier




# Create and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


from sklearn.svm import SVC

# Create and train the SVM model
model = SVC()
model.fit(X_train, y_train)


from sklearn.naive_bayes import GaussianNB

# Create and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier

# Create and train the KNN model
model = KNeighborsClassifier(n_neighbors=3)  # Adjust n_neighbors as needed
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.ensemble import GradientBoostingClassifier

# Create and train the GBM model
model = GradientBoostingClassifier(n_estimators=100)  # Adjust n_estimators as needed
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


import xgboost as xgb

# Create and train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


import lightgbm as lgb

# Create and train the LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from catboost import CatBoostClassifier

# Create and train the CatBoost model
model = CatBoostClassifier(verbose=False)
model.fit(X_train, y_train)

# Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)