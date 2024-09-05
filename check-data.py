import pandas as pd

dataset = pd.read_csv('data/mushroom_raw.csv')
print(dataset.head())

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

# Check distribution of variables
import matplotlib.pyplot as plt

dataset.hist(bins=50, figsize=(16, 9))
plt.show()

# Save cleaned dataset
dataset.to_csv('data/mushroom_cleaned.csv', index=False)