# house_price_prediction
# House price prediction is a process of estimating the market value of residential properties based on various features such as location, size, number of bedrooms, amenities, and historical price trends. 
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20, 10)

# Importing the dataset
dataset = pd.read_csv('../dataset/Bengaluru_House_Data.csv')
print(dataset.head(10))
print(dataset.shape)

# Data Preprocessing
# Getting the count of area type in the dataset
print(dataset.groupby('area_type')['area_type'].agg('count'))

# Dropping unnecessary columns
dataset.drop(['area_type', 'society', 'availability', 'balcony'], axis='columns', inplace=True)
print(dataset.shape)
# Data cleaning
print(dataset.isnull().sum())
dataset.dropna(inplace=True)
print(dataset.shape)

# Data engineering
print(dataset['size'].unique())
dataset['bhk'] = dataset['size'].apply(lambda x: float(x.split(' ')[0]))

# Exploring 'total_sqft' column
print(dataset['total_sqft'].unique())

# Defining a function to check whether the value is float or not
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

print(dataset[~dataset['total_sqft'].apply(is_float)].head(10))

# Defining a function to convert range values to a single value
print(convert_sqft_to_num('290'))
print(convert_sqft_to_num('2100 - 2850'))
print(convert_sqft_to_num('4.46Sq. Meter'))

# Applying this function to the dataset
dataset['total_sqft'] = dataset['total_sqft'].apply(convert_sqft_to_num)
print(dataset['total_sqft'].head(10))
print(dataset.loc[30])

# Feature engineering
print(dataset.head(10))

# Creating a new column 'price_per_sqft'
dataset['price_per_sqft'] = dataset['price'] * 100000 / dataset['total_sqft']
print(dataset['price_per_sqft'])

# Exploring 'location' column
print(len(dataset['location'].unique()))
dataset['location'] = dataset['location'].apply(lambda x: x.strip())
location_stats = dataset.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats[0:10])

# Creating 'location_stats_less_than_10' for locations with ≤10 occurrences
location_stats_less_than_10 = location_stats[location_stats <= 10]
dataset['location'] = dataset['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(dataset['location'].head(10))
print(len(dataset['location'].unique()))

# Outlier detection and removal
print(dataset[dataset['total_sqft'] / dataset['bhk'] < 300].sort_values(by='total_sqft').head(10))
dataset = dataset[~(dataset['total_sqft'] / dataset['bhk'] < 300)]
print(dataset.shape)

# Removing outliers in 'price_per_sqft'
print(dataset['price_per_sqft'].describe())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean = np.mean(subdf['price_per_sqft'])
        std = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (mean - std)) & (subdf['price_per_sqft'] <= (mean + std))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

dataset = remove_pps_outliers(dataset)
print(dataset.shape)

# Defining a function to remove BHK outliers
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }

for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df['price_per_sqft'] < stats['mean']].index.values
                )
    return df.drop(exclude_indices, axis='index')

dataset = remove_bhk_outliers(dataset)
print(dataset.shape)

# Dropping unwanted features
dataset.drop(['size', 'price_per_sqft'], axis='columns', inplace=True)
print(dataset.head())

# One-hot encoding 'location' column
dummies = pd.get_dummies(dataset['location'])
dataset = pd.concat([dataset, dummies.drop('other', axis='columns')], axis='columns')
dataset.drop('location', axis=1, inplace=True)
print(dataset.head())
print(dataset.shape)

# Splitting the dataset into independent (X) and dependent (y) variables
X = dataset.drop(['price'], axis='columns')
y = dataset['price']
print(X.shape)
print(y.shape)
# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))

# K-fold cross-validation
from sklearn.model_selection import ShuffleSplit, cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print(cross_val_score(regressor, X, y, cv=cv))

# Hyperparameter tuning using Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearch(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {'normalize': [True, False]}
        },
        'lasso': {
            'model': Lasso(),
            'params': {'alpha': [1, 2], 'selection': ['random', 'cyclic']}
 },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {'criterion': ['mse', 'friedman_mse'], 'splitter': ['best', 'random']}
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, n_jobs=-1, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

model_scores = find_best_model_using_gridsearch(X, y)
print(model_scores)

# Using the best model (Linear Regression)
regressor.fit(X, y)

# Prediction function
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return regressor.predict([x])[0]

print(predict_price('1st Phase JP Nagar', 1000, 2, 2))
print(predict_price('Indira Nagar', 1000, 3, 3))

# Saving the model
import pickle
with open('bangalore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(regressor, f)

# Exporting columns
import json
columns = {'data_columns': [col.lower() for col in X.columns]}
with open("columns.json", "w") as f:
