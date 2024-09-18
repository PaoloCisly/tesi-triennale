from alive_progress import alive_bar

from utils import Dataset, Model, load_grid_config

def train(dataset: Dataset, X_train: list, y_train: list) -> tuple:
    """Train the best models found by grid search.

    Parameters
    ----------
    dataset: Dataset
        The dataset used to find the best parameters.
    X_train: list
        The training set.
    y_train: list
        The labels.

    Returns
    -------
    tuple
        A tuple containing the training times list and the trained models list.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    import time

    print('Training best models...')
    
    models = [KNeighborsClassifier, GaussianNB, MLPClassifier, RandomForestClassifier, SVC]

    best_params = [load_grid_config(dataset, model) for model in Model]
    train_times = []
    trained_models = []

    with alive_bar(len(models)) as bar:
        for model, params in zip(models, best_params):
            train_time = time.time()
            model = model(**params)
            model.fit(X_train, y_train)
            train_time = time.time() - train_time

            train_times.append(train_time)
            trained_models.append(model)
            bar()

    return train_times, trained_models

def prediction(X_test: list, best_models: list) -> tuple:
    """Predict the test set using the best models.

    Parameters
    ----------
    X_test: list
        The test set.
    best_models: list
        The trained models.

    Returns
    -------
    tuple
        A tuple containing the prediction times list and the predictions list.
    """
    import time

    print('Making predictions...')

    pred_times = []
    predictions = []

    with alive_bar(len(best_models)) as bar:
        for model in best_models:
            exec_time = time.time()
            y_pred = model.predict(X_test)
            pred_times.append(time.time() - exec_time)
            predictions.append(y_pred)
            bar()

    return pred_times, predictions

def learning_curves(best_models: list, X_train: list, y_train: list, dataset: Dataset, force: bool = False) -> tuple:
    """Calculate the learning curves for the best models.

    Parameters
    ----------
    best_models: list
        The trained models.
    X_train: list
        The training set.
    y_train: list
        The labels.

    Returns
    -------
    tuple
        A tuple containing the training sizes list, the training scores list, and the test scores list.
    """
    import os

    if not os.path.exists('./data/learning_curves'):
        os.makedirs('./data/learning_curves')

    if not force and os.path.exists(f'./data/learning_curves/{dataset.name}.pkl'):
        import pickle
        print('Importing learning curves...')
        with open(f'./data/learning_curves/{dataset.name}.pkl', 'rb') as f:
            return pickle.load(f)

    print('Calculating learning curves...')

    from sklearn.model_selection import learning_curve
    import numpy as np

    train_sizes_list = []
    train_scores_list = []
    test_scores_list = []

    with alive_bar(len(best_models)) as bar:
        for model in best_models:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 5))

            train_sizes_list.append(train_sizes)
            train_scores_list.append(train_scores)
            test_scores_list.append(test_scores)
            bar()

    with open(f'./data/learning_curves/{dataset.name}.pkl', 'wb') as f:
        import pickle
        pickle.dump([train_sizes_list, train_scores_list, test_scores_list], f)

    return [train_sizes_list, train_scores_list, test_scores_list]
