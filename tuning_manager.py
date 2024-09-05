from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

import model_tuning

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
knn_best_params, knn_best_score = model_tuning.knn_search(X_train, y_train)