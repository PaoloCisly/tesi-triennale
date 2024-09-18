from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

import model_tuning, utils

def run(dataset_type: utils.Dataset):
    X_train, X_test, y_train, y_test = utils.load_dataset(dataset_type)

    knn_time = time.time()
    # K-Nearest Neighbors (K-NN)
    knn_best_params = model_tuning.knn_search(dataset_type, X_train, y_train)
    knn_time = time.time() - knn_time

    nb_time = time.time()
    # Naive Bayes
    nb_best_params = model_tuning.nb_search(dataset_type, X_train, y_train)
    nb_time = time.time() - nb_time

    nn_time = time.time()
    # Neural Network
    nn_best_params = model_tuning.nn_search(dataset_type, X_train, y_train)
    nn_time = time.time() - nn_time

    rf_time = time.time()
    # Random Forest
    rf_best_params = model_tuning.rf_search(dataset_type, X_train, y_train)
    rf_time = time.time() - rf_time

    svm_time = time.time()
    # Support Vector Machine (SVM)
    svm_best_params = model_tuning.svm_search(dataset_type, X_train, y_train)
    svm_time = time.time() - svm_time

    # xgb_time = time.time()
    # # XGBoost
    # xgb_best_params = model_tuning.xgb_search(dataset_type, X_train, y_train)
    # xgb_time = time.time() - xgb_time

    print(f'K-Nearest Neighbors Best Parameters: \n{knn_best_params}')
    print(f'K-Nearest Neighbors Time: {knn_time:.2f}s\n\n')

    print(f'Naive Bayes Best Parameters: \n{nb_best_params}')
    print(f'Naive Bayes Time: {nb_time:.2f}s\n\n')

    print(f'Neural Network Best Parameters: \n{nn_best_params}')
    print(f'Neural Network Time: {nn_time:.2f}s\n\n')

    print(f'Random Forest Best Parameters: \n{rf_best_params}')
    print(f'Random Forest Time: {rf_time:.2f}s\n\n')

    print(f'Support Vector Machine Best Parameters: \n{svm_best_params}')
    print(f'Support Vector Machine Time: {svm_time:.2f}s\n\n')

    # print(f'XGBoost Best Parameters: \n{xgb_best_params}')
    # print(f'XGBoost Time: {xgb_time:.2f}s\n\n')

