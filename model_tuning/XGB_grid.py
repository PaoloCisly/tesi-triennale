import os.path

import utils

def xgb_search(dataset, X_train, y_train, force = False):
    if not force and os.path.isfile(f'./data/grid_configs/{dataset.name}_{utils.Model.XGB.name}.pkl'):
        return utils.load_grid_config(dataset, utils.Model.XGB)
    
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier

    xgb = XGBClassifier()
    parameters = {
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    grid_search = GridSearchCV(estimator=xgb,
                                param_grid=parameters,
                                scoring='accuracy',
                                cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f'XGBoost Best Parameters: {grid_search.best_params_}')
    print(f'XGBoost Best Score: {grid_search.best_score_}')

    utils.save_grid_config(dataset, utils.Model.XGB, grid_search.best_params_)

    return grid_search.best_params_, grid_search.best_score_