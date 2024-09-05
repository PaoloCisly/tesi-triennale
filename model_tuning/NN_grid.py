def nn_search(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPClassifier

    nn = MLPClassifier()
    parameters = {
        'hidden_layer_sizes': [(100,), (200,), (300,), (400,), (500,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    grid_search = GridSearchCV(estimator=nn,
                                param_grid=parameters,
                                scoring='accuracy',
                                cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f'Neural Network Best Parameters: {grid_search.best_params_}')
    print(f'Neural Network Best Score: {grid_search.best_score_}')

    return grid_search.best_params_, grid_search.best_score_