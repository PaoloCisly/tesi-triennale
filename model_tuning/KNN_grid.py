import os.path

import utils

def knn_search(dataset, X_train, y_train, force = False):
    if not force and os.path.isfile(f'./data/grid_configs/{dataset.name}_{utils.Model.KNN.name}.pkl'):
        return utils.load_grid_config(dataset, utils.Model.KNN)

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
                                cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f'K-NN Best Parameters: {grid_search.best_params_}')
    print(f'K-NN Best Score: {grid_search.best_score_}')

    utils.save_grid_config(dataset, utils.Model.KNN, grid_search.best_params_)

    return grid_search.best_params_