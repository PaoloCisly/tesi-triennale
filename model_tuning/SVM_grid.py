def svm_search(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    svm = SVC()
    parameters = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [1, 2, 3, 4, 5],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(estimator=svm,
                                param_grid=parameters,
                                scoring='accuracy',
                                cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f'SVM Best Parameters: {grid_search.best_params_}')
    print(f'SVM Best Score: {grid_search.best_score_}')

    return grid_search.best_params_, grid_search.best_score_