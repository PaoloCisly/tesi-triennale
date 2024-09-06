from enum import Enum

class Dataset(Enum):
    MUSHROOM = 1
    
class Model(Enum):
    KNN = 1
    NB = 2
    NN = 3
    RF = 4
    SVM = 5
    XGB = 6

def save_grid_config(dataset, model, params):
    import pickle 
    
    with open(f'{dataset.name}_{model.name}_grid_config.pkl', 'wb') as f:
        pickle.dump(dictionary, f)

def load_grid_config(dataset, model):
    import pickle
    
    with open(f'{dataset.name}_{model.name}_grid_config.pkl', 'rb') as f:
        params = pickle.load(f)
        
    return params
    