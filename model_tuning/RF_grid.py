import os.path

import utils

def rf_search(dataset, X_train, y_train, force = False):
    if not force and os.path.isfile(f'./data/grid_configs/{dataset.name}_{utils.Model.RF.name}.pkl'):
        return utils.load_grid_config(dataset, utils.Model.RF)
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier()
    parameters = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=rf,
                                param_grid=parameters,
                                scoring='accuracy',
                                cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f'Random Forest Best Parameters: {grid_search.best_params_}')
    print(f'Random Forest Best Score: {grid_search.best_score_}')

    utils.save_grid_config(dataset, utils.Model.RF, grid_search.best_params_)

    return grid_search.best_params_, grid_search.best_score_