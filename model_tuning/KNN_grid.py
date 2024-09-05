def knn_search(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    parameters = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['cosine', 'euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(estimator=knn,
                                param_grid=parameters,
                                scoring='accuracy',
                                cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f'K-NN Best Parameters: {grid_search.best_params_}')
    print(f'K-NN Best Score: {grid_search.best_score_}')

    return grid_search.best_params_, grid_search.best_score_