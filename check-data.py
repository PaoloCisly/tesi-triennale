import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

print('''Seleziona il dataset da pulire:
1. Mushroom Dataset
2. Hotel Reservations Dataset
3. Weather Type Classification
4. Star Type Classification
5. Esci\n''')

while True:
    try:
        option = int(input(">> "))
        if option == 1:
            dataset_name = 'mushroom'
            break
        elif option == 2:
            dataset_name = 'hotel'
            break
        elif option == 3:
            dataset_name = 'weather'
            break
        elif option == 4:
            dataset_name = 'star'
            break
        elif option == 5:
            exit()
        else:
            print("Inserisci un valore valido")
    except ValueError:
        print("Inserisci un valore valido")

# Load dataset
dataset = pd.read_csv(f'data/raw_datasets/{dataset_name}.csv', encoding = 'unicode_escape')
print(dataset.head())

# Remove first column of IDs
if dataset_name == 'hotel' or dataset_name == 'weather':
    dataset = dataset.iloc[:, 1:]

# Describe dataset
print('\n\nDescribe dataset:')
print(dataset.describe())

# Check for missing values
print('\n\nCheck for missing values:')
print(dataset.isnull().sum())
# Drop missing values
dataset.dropna(inplace=True)

# Check for duplicate values
print('\n\nCheck for duplicate values:')
print(f'Duplicates Raw: \t{dataset.duplicated().sum()}')
# Drop duplicates
dataset.drop_duplicates(inplace=True)
print(f'Duplicates Cleaned:\t{dataset.duplicated().sum()}')

# Check type of each column
print('\n\nCheck type of each column:')
print(dataset.dtypes)

# Check number of rows
print('\n\nCheck number of rows:')
print(f'Number of rows: {dataset.shape[0]}')

# Check distribution of target variable
print('\n\nCheck distribution of target variable:')
print(dataset['class'].value_counts())

# Encoding of categorical variables
label_encoders = {}
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
        label_encoders[column] = le

# Check distribution of variables
dataset.hist(bins=50, figsize=(16, 9))
plt.show()

# Save cleaned dataset
dataset.to_csv(f'data/{dataset_name}_cleaned.csv', index=False)