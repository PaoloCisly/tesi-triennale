from enum import Enum
class Dataset(Enum):
    """Enum class for the available datasets types."""
    MUSHROOM = 'data/mushroom_cleaned.csv'
    HOTEL = 'data/hotel_cleaned.csv'
    WEATHER = 'data/weather_cleaned.csv'
    STAR = 'data/star_cleaned.csv'
    
class Model(Enum):
    """Enum class for the available models types."""
    KNN = 0
    NB = 1
    NN = 2
    RF = 3
    SVM = 4

def load_dataset(dataset: Dataset) -> tuple:
    """Load the cleaned dataset and split it into training and testing sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset type to load.
    
    Returns
    -------
    tuple
        A tuple containing the training and testing sets.
    """

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    print(f'Loading dataset {dataset.name}...')

    dataset = pd.read_csv(dataset.value)

    # Splitting the dataset into the Training set and Test set
    X = dataset.drop('class', axis=1)
    y = dataset['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print('Dataset loaded successfully!')

    return X_train, X_test, y_train, y_test

def save_grid_config(dataset: Dataset, model: Model, params: dict) -> None:
    """Save the best parameters found by grid search.

    Parameters
    ----------
    dataset: Dataset
        The dataset used to find the best parameters.
    model: Model
        The model used to find the best parameters.
    params: dict
        The best parameters found by grid search.
    """
    import pickle, os

    if not os.path.exists('./data/grid_configs'):
        os.makedirs('./data/grid_configs')
    with open(f'./data/grid_configs/{dataset.name}_{model.name}.pkl', 'wb') as f:
        pickle.dump(params, f)

def load_grid_config(dataset: Dataset, model: Model) -> dict:
    """Load the best parameters found by grid search.

    Parameters
    ----------
    dataset: Dataset
        The dataset used to find the best parameters.
    model: Model
        The model used to find the best parameters.

    Returns
    -------
    dict
        The best parameters found by grid search.
    """
    import pickle
    
    with open(f'./data/grid_configs/{dataset.name}_{model.name}.pkl', 'rb') as f:
        params = pickle.load(f)
        
    return params

def get_centered_dataset_name(dataset_name: str):
    total_length = 53
    side_length = (total_length - len(dataset_name) - 2) // 2
    side_length_right = side_length + (len(dataset_name) % 2 == 0)
    return f"{'-' * side_length} {dataset_name} {'-' * side_length_right}"

def print_results(results: dict) -> None:
    """Print the results of the models for one dataset.

    Parameters
    ----------
    results: dict
        The results of the models.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    for model_type in results:
        print(f'{model_type}:')
        print('Accuracy:', results[model_type]['accuracy'])
        print('Classification Report:')
        print(classification_report(results[model_type]['y_test'], results[model_type]['y_preds']))
        print('Confusion Matrix:')
        print(confusion_matrix(results[model_type]['y_test'], results[model_type]['y_preds']))
        print('Train Time:', results[model_type]['train_time'])
        print('Prediction Time:', results[model_type]['pred_time'])
        print()

def save_results(results: dict) -> None:
    """Save the results of the models for all datasets.

    Parameters
    ----------
    results: dict
        The results of the models.
    """
    import os, time
    from sklearn.metrics import classification_report, confusion_matrix
    
    if not os.path.exists('./data/results'):
        os.makedirs('./data/results')

    with open(f'./data/results/results_{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f:
        for dataset in results:
            f.write(f'{get_centered_dataset_name(dataset)}\n\n')
            for model_type in results[dataset]:
                f.write(f'{model_type}:\n')
                f.write(f'Accuracy: {results[dataset][model_type]["accuracy"]}\n')
                f.write('Classification Report:\n')
                f.write(f'{classification_report(results[dataset][model_type]["y_test"], results[dataset][model_type]["y_preds"])}\n')
                f.write('Confusion Matrix:\n')
                f.write(f'{confusion_matrix(results[dataset][model_type]["y_test"], results[dataset][model_type]["y_preds"])}\n')
                f.write(f'Train Time: {results[dataset][model_type]["train_time"]}\n')
                f.write(f'Prediction Time: {results[dataset][model_type]["pred_time"]}\n')
                f.write('\n')
            f.write('\n\n')

def check_tuning_status() -> list:
    """Check the tuning status of the datasets.
    
    Returns
    -------
    list
        A list of datasets that need tuning."""
    
    import os
    from colorama import Fore, init
    init(autoreset=True)

    datasets_to_tuning = [dataset for dataset in Dataset]
    max_length = max(len(dataset.name) for dataset in Dataset)

    print(f'{Fore.WHITE}\nDatasets Tuning Status:\n')
    
    if os.path.exists('./data/grid_configs'):
        for dataset in Dataset:
            status = True
            for model in Model:
                if not os.path.exists(f'./data/grid_configs/{dataset.name}_{model.name}.pkl'):
                    status = False
                    break
            if status:
                print(f'{Fore.WHITE}{dataset.name.ljust(max_length)}\t[{Fore.GREEN}\u2713{Fore.WHITE}]')
                datasets_to_tuning.remove(dataset)
            else:
                print(f'{Fore.WHITE}{dataset.name.ljust(max_length)}\t[{Fore.RED}\u2717{Fore.WHITE}]')
    else:
        os.makedirs('./data/grid_configs')
        for dataset in Dataset:
            print(f'{Fore.WHITE}{dataset.name.ljust(max_length)}\t[{Fore.RED}\u2717{Fore.WHITE}]')
    
    print()
    return datasets_to_tuning

def make_plots(dataset: Dataset, results: dict) -> None:
    """Make theplots for accuracy, learning curves and time performance for the best model.

    Parameters
    ----------
    dataset: Dataset
        The dataset used to find the best parameters.
    results: dict
        The results of the model.
    """
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    import numpy as np

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), [results[model]['accuracy'] for model in results], color='mediumseagreen')
    plt.title(f'Accuracy for {dataset.name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.savefig(f'./images/{dataset.name}_accuracy.png')

    # Time train plot
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), [results[model]['train_time'] for model in results], color='turquoise')
    plt.title(f'Train Time for {dataset.name}')
    plt.ylabel('Time (s)')
    plt.xlabel('Model')
    plt.savefig(f'./images/{dataset.name}_train_time.png')

    # Time prediction plot
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), [results[model]['pred_time'] for model in results], color='coral')
    plt.title(f'Prediction Time for {dataset.name}')
    plt.ylabel('Time (s)')
    plt.xlabel('Model')
    plt.savefig(f'./images/{dataset.name}_pred_time.png')

    # Learning curves plot
    for model in results:
        train_mean = np.mean(results[model]['train_scores'], axis=1)
        train_std = np.std(results[model]['train_scores'], axis=1)
        test_mean = np.mean(results[model]['test_scores'], axis=1)
        test_std = np.std(results[model]['test_scores'], axis=1)

        plt.figure(figsize=(10, 5))
        plt.plot(results[model]['train_sizes'], train_mean, label='Train score', color='turquoise')
        plt.plot(results[model]['train_sizes'], test_mean, label='Test score', color='coral')

        plt.fill_between(results[model]['train_sizes'], train_mean - train_std, train_mean + train_std, color='turquoise', alpha=0.1)
        plt.fill_between(results[model]['train_sizes'], test_mean - test_std, test_mean + test_std, color='coral', alpha=0.1)

        plt.title(f'Learning Curves for {dataset.name} - {model}')
        plt.ylabel('Score')
        plt.xlabel('Training examples')
        plt.legend()
        plt.savefig(f'./images/{dataset.name}_{model}_learning_curve.png')

