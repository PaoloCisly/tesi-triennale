from colorama import Fore, init
init(autoreset=True)
print(f'{Fore.WHITE}Data1:\t\t[{Fore.GREEN}\u2713{Fore.WHITE}]')
print(f'{Fore.WHITE}Data2:\t\t[{Fore.RED}\u2717{Fore.WHITE}]')

print('\n\n\n\n')

from utils import Dataset, Model
def check_tuning_status() -> list:
    
    import os
    from colorama import Fore, init
    init(autoreset=True)

    datasets_to_tuning = [dataset for dataset in Dataset]
    max_length = max(len(dataset.name) for dataset in Dataset)

    print(f'{Fore.WHITE}Dataset Tuning Status:\n')
    
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
        for dataset in Dataset:
            print(f'{Fore.WHITE}{dataset.name.ljust(max_length)}\t[{Fore.RED}\u2717{Fore.WHITE}]')
    
    print()
    return datasets_to_tuning

print('\n\n')
print(check_tuning_status())
print('\n\n\n\n')

import time
import sys

# Timer che cambia sulla stessa riga
for i in range(10, 0, -1):
    sys.stdout.write('\r')
    sys.stdout.write(f'Avvio TUNING per tutti i dataset mancanti in {i} secondi...')
    sys.stdout.flush()
    time.sleep(1)

print(f'\nAvvio...')