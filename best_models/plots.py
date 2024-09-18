from utils import Model

def make_plots(models_results: dict) -> None:
    """Make plots to compare the models.

    Parameters
    ----------
    models_results: dict
        A dictionary containing the results of the models.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    model_names = [model.name for model in Model]
    accuracies = [models_results[model]['accuracy'] for model in model_names]

    # Plotting the accuracies
    plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('Modelli')
    plt.ylabel('Accuracy')
    plt.title('Confronto Accuracy Modelli Ottimizzati')
    plt.show()

    # Plotting the training times
    train_times = [models_results[model]['train_time'] for model in model_names]
    plt.bar(model_names, train_times, color='lightgreen')
    plt.title('Tempo di Addestramento per Modello')
    plt.ylabel('Tempo (s)')
    plt.show()

    # Plotting the prediction times
    pred_times = [models_results[model]['pred_time'] for model in model_names]
    plt.bar(model_names, pred_times, color='lightblue')
    plt.title('Tempo di Predizione per Modello')
    plt.ylabel('Tempo (s)')
    plt.show()

    # Plotting the learning curves
    for model in model_names:
        train_sizes = models_results[model]['train_sizes']
        train_scores = models_results[model]['train_scores']
        test_scores = models_results[model]['test_scores']

        # Calculate mean and standard deviation for the curves
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plotting the learning curve
        plt.plot(train_sizes, train_mean, label="Training score", color="blue")
        plt.plot(train_sizes, test_mean, label="Cross-validation score", color="green")

        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="green", alpha=0.1)

        plt.title(f'Curva di Apprendimento per {model}')
        plt.xlabel('Numero di campioni di training')
        plt.ylabel('Accuracy')
        plt.legend(loc="best")
        plt.show()
