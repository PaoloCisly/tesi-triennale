import os.path

import utils

def nn_search(dataset, X_train, y_train, force = False):
    if not force and os.path.isfile(f'./data/grid_configs/{dataset.name}_{utils.Model.NN.name}.pkl'):
        return utils.load_grid_config(dataset, utils.Model.NN)

    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPClassifier

    nn = MLPClassifier()
    parameters = {
        'hidden_layer_sizes': [(50,), (100,), (200,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }

    grid_search = GridSearchCV(estimator=nn,
                                param_grid=parameters,
                                scoring='accuracy',
                                cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f'Neural Network Best Parameters: {grid_search.best_params_}')
    print(f'Neural Network Best Score: {grid_search.best_score_}')

    utils.save_grid_config(dataset, utils.Model.NN, grid_search.best_params_)

    return grid_search.best_params_, grid_search.best_score_