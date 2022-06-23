#%%
# Imports and Data Read
from cgi import test
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



import warnings
warnings.simplefilter("ignore")

filepath = 'Data/food_delivery/cleaned_full_data.csv'
data = pd.read_csv(filepath)

#%%
# Data Cleaning

data_clean = data.copy()
data_clean = data_clean.drop(['Unnamed: 0', 'url','distance','category_2'], axis=1)

for i in range(len(data_clean)):
    if data_clean['price_range'][i] == '$':
        data_clean['price_range'][i] = 1
    elif data_clean['price_range'][i] == '$$':
        data_clean['price_range'][i] = 2
    elif data_clean['price_range'][i] == '$$$':
        data_clean['price_range'][i] = 3
    elif data_clean['price_range'][i] == '$$$$':
        data_clean['price_range'][i] = 4

data_clean.dropna(axis=0,inplace=True)
data_clean['price_range'] = pd.to_numeric(data_clean['price_range'])

data_clean_final = pd.get_dummies(data_clean, columns=['category_1'], drop_first=True)
data_clean_final.columns = data_clean_final.columns.str.replace('category_1_', '')
data_clean.reset_index()
data_clean_final.reset_index()

# %%
# Visualization 
fig1 = plt.figure(figsize=(20,10))
ax1 = plt.axes()
ax1 = plt.hist(data_clean_final['price_range'], bins=4)
plt.xticks([1,2,3,4])

# %%
fig2 = plt.figure(figsize=(50,10))
ax2 = plt.axes()
ax2 = plt.hist(data_clean['category_1'], bins=274)
plt.xticks(
    rotation=90,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small',
)
# %%
y_col = 'price_range'
feature_cols = [x for x in data_clean.columns if x != y_col]
X_data = data_clean[feature_cols]
y_data = data_clean[y_col]
feature_cols = [x for x in data_clean_final.columns if x != y_col]
X_data_ohc = data_clean_final[feature_cols]
y_data_ohc = data_clean_final[y_col]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X_data_ohc, y_data_ohc, test_size=0.3, random_state=42)

# Compare the indices to ensure they are identical
(X_train_ohc.index == X_train.index).all()

#%%

#%% 
pd.options.mode.chained_assignment = None

scalers = {'standard': StandardScaler(),
           'minmax': MinMaxScaler(),
           'maxabs': MaxAbsScaler()}

training_test_sets = {
    'not_encoded': (X_train, y_train, X_test, y_test),
    'one_hot_encoded': (X_train_ohc, y_train_ohc, X_test_ohc, y_test_ohc)}


# Get the list of float columns, and the float data
# so that we don't scale something we already scaled. 
# We're supposed to scale the original data each time
mask = X_train.dtypes == np.float
float_columns = X_train.columns[mask]

# initialize model
LR = LinearRegression()

# iterate over all possible combinations and get the errors
errors = {}
for encoding_label, (_X_train, _y_train, _X_test, _y_test) in training_test_sets.items():
    for scaler_label, scaler in scalers.items():
        trainingset = _X_train.copy()  # copy because we dont want to scale this more than once.
        testset = _X_test.copy()
        trainingset[float_columns] = scaler.fit_transform(trainingset[float_columns])
        testset[float_columns] = scaler.transform(testset[float_columns])
        LR.fit(trainingset, _y_train)
        predictions = LR.predict(testset)
        key = encoding_label + ' - ' + scaler_label + 'scaling'
        errors[key] = mean_squared_error(_y_test, predictions)

errors = pd.Series(errors)
print(errors.to_string())
print('-' * 80)
for key, error_val in errors.items():
    print(key, error_val)

# %%
print(X_train.dtypes)
# %%
