def nb_search(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import GaussianNB

    nb = GaussianNB()
    parameters = {
        'var_smoothing': [1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    }

    grid_search = GridSearchCV(estimator=nb,
                                param_grid=parameters,
                                scoring='accuracy',
                                cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f'Naive Bayes Best Parameters: {grid_search.best_params_}')
    print(f'Naive Bayes Best Score: {grid_search.best_score_}')

    return grid_search.best_params_, grid_search.best_score_