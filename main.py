from sklearn.metrics import accuracy_score
import os, sys, time

import utils, best_models, tuning_manager

datasets_to_tuning = utils.check_tuning_status()

if datasets_to_tuning:
    for i in range(10, 0, -1):
        sys.stdout.write('\r')
        sys.stdout.write(f'Avvio TUNING per tutti i dataset mancanti in {i} secondi...')
        sys.stdout.flush()
        time.sleep(1)

    os.system('cls' if os.name == 'nt' else 'clear')

    print(f'\nAvvio...')
    for dataset in datasets_to_tuning:
        tuning_manager.run(dataset)

    os.system('cls' if os.name == 'nt' else 'clear')

else:
    print('TUNING già completato per tutti i dataset\n')
    time.sleep(4)
    os.system('cls' if os.name == 'nt' else 'clear')

print(''' ____        _                 _        _                _                    
|  _ \  __ _| |_ __ _ ___  ___| |_     / \   _ __   __ _| |_   _ _______ _ __ 
| | | |/ _` | __/ _` / __|/ _ \ __|   / _ \ | '_ \ / _` | | | | |_  / _ \ '__|
| |_| | (_| | || (_| \__ \  __/ |_   / ___ \| | | | (_| | | |_| |/ /  __/ |   
|____/ \__,_|\__\__,_|___/\___|\__| /_/   \_\_| |_|\__,_|_|\__, /___\___|_|   
                                                           |___/              ''')

print('''Seleziona il dataset da analizzare:\n
0. Tutti i dataset
1. Mushroom Dataset
2. Rain in Australia
3. Weather Type Classification
4. Star Type Classification
5. Esci\n''')

while True:
    try:
        option = int(input(">> "))
        if option == 0:
            datasets = [utils.Dataset.MUSHROOM, utils.Dataset.HOTEL, utils.Dataset.WEATHER, utils.Dataset.STAR]
            break
        elif option == 1:
            datasets = [utils.Dataset.MUSHROOM]
            break
        elif option == 2:
            datasets = [utils.Dataset.HOTEL]
            break
        elif option == 3:
            datasets = [utils.Dataset.WEATHER]
            break
        elif option == 4:
            datasets = [utils.Dataset.STAR]
            break
        elif option == 5:
            exit()
        else:
            print("Inserisci un valore valido")
    except ValueError:
        print("Inserisci un valore valido")

os.system('cls' if os.name == 'nt' else 'clear')

results = {}

for dataset in datasets:
    print(f'\nAnalisi del dataset {dataset.name} in corso...\n')
    # Load dataset
    X_train, X_test, y_train, y_test = utils.load_dataset(dataset)
    # Train best models
    train_times, best_models_list = best_models.train(dataset, X_train, y_train)
    # Make predictions
    pred_times, y_preds = best_models.prediction(X_test, best_models_list)
    # Make learning curves
    train_sizes, train_scores, test_scores = best_models.learning_curves(best_models_list, X_train, y_train, dataset)
    
    # Save metrics
    models_results = {}
    for i, model_type in enumerate(utils.Model):
        models_results[model_type.name] = {
            'y_test': y_test,
            'y_preds': y_preds[i],
            'accuracy': accuracy_score(y_test, y_preds[i]),
            'train_time': train_times[i],
            'pred_time': pred_times[i],
            'train_sizes': train_sizes[i],
            'train_scores': train_scores[i],
            'test_scores': test_scores[i]
        }

    utils.print_results(models_results)
    results[dataset.name] = models_results
    utils.make_plots(dataset, models_results)

    print(f'Analisi del dataset {dataset.name} completata\n')

utils.save_results(results)