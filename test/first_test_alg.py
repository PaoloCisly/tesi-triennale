from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importing the dataset
import pandas as pd

dataset = pd.read_csv('data/mushroom_cleaned.csv')

# Splitting the dataset into the Training set and Test set
X = dataset.drop('class', axis=1)
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Nearest Neighbors (K-NN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f'K-NN Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'K-NN Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'K-NN Classification Report:\n{classification_report(y_test, y_pred)}')
print('\n')

# Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(f'SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}')
print(f'SVM Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svm)}')
print(f'SVM Classification Report:\n{classification_report(y_test, y_pred_svm)}')
print('\n')

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Random Forest Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}')
print(f'Random Forest Classification Report:\n{classification_report(y_test, y_pred_rf)}')
print('\n')

# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f'XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}')
print(f'XGBoost Confusion Matrix:\n{confusion_matrix(y_test, y_pred_xgb)}')
print(f'XGBoost Classification Report:\n{classification_report(y_test, y_pred_xgb)}')
print('\n')

# Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
print(f'Neural Network Accuracy: {accuracy_score(y_test, y_pred_nn)}')
print(f'Neural Network Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nn)}')
print(f'Neural Network Classification Report:\n{classification_report(y_test, y_pred_nn)}')
print('\n')

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(f'Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb)}')
print(f'Naive Bayes Confusion Matrix:\n{confusion_matrix(y_test, y_pred_nb)}')
print(f'Naive Bayes Classification Report:\n{classification_report(y_test, y_pred_nb)}')
print('\n')

