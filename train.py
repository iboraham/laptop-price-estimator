import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv('data.csv')



def _handle_cpu_speed(value):
    if pd.isna(value):
        return np.nan

    # Remove non-numeric characters
    value = ''.join(filter(str.isdigit, str(value)))

    # Handle cases with multiple values, in this case, taking the average
    if ',' in value:
        values = value.split(',')
        value = sum(map(float, values)) / len(values)

    # Convert to float and convert GHz to MHz (if needed)
    if 'GHz' in str(value):
        value = float(value.replace('GHz', '')) * 1000
    else:
        value = float(value)
    
    return value

def preprocess(data:pd.DataFrame)->pd.DataFrame:
    """
    This function preprocesses the input dataframe by performing the following operations in order:
        - Removing any duplicate rows
        - Dropping columns with too many missing values
        - Converting categorical columns to one hot encoding
        - Trimming some columns to convert them to numeric values e.g. removing the $ sign from the price column
        - Removing any rows with missing values

    Columns:
        brand: The brand of the laptop (OHE)
        model: The model of the laptop (OHE)
        screen_size: The screen size of the laptop (numeric), needs trimming e.g. 15.6 inches -> 15.6
        color: The color of the laptop (OHE)
        harddisk: The size of the harddisk (numeric), needs trimming e.g. 1000 GB -> 1000
        cpu: The CPU of the laptop (OHE)
        ram: The size of the RAM (numeric), needs trimming e.g. 8 GB -> 8
        OS: The operating system of the laptop (OHE)
        special_features: The special features of the laptop (OHE)
        graphics: If the graphics card is dedicated or integrated (Convert to 0/1)
        graphics_coprocessor: The graphics card of the laptop (OHE)
        cpu_speed: The speed of the CPU (numeric), needs trimming e.g. 2.3 GHz -> 2.3
        rating: The rating of the laptop (numeric)
        price: The price of the laptop (numeric), needs trimming e.g. $1,000 -> 1000 or $1,599.99 -> 1599.99

    New columns can be created:
        cpu_brand: The brand of the CPU (OHE) e.g. cpu = Intel Core i5 -> cpu_brand = Intel
        x

    
    Parameters:
    data (pd.DataFrame): The input dataframe to be preprocessed
    
    Returns:
    pd.DataFrame: The preprocessed dataframe
    """

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Drop columns with too many missing values
    missing_cols = [col for col in data.columns if data[col].isnull().mean() > 0.1]
    data = data.drop(missing_cols, axis=1)

    # Create new columns
    data['cpu_brand'] = data['cpu'].str.split().str[0]
    # data['cpu_gen'] = data['cpu'].str.split().str[2].astype(float)

    # Convert categorical columns to one hot encoding
    categorical_cols = ["brand", "model", "color", "cpu", "OS", "special_features", "graphics_coprocessor", "cpu_brand"]
    categorical_cols = [col for col in categorical_cols if col in data.columns]
    encoder = OneHotEncoder()
    data_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_cols]).toarray(), columns=encoder.get_feature_names_out(categorical_cols))
    data = pd.concat([data.drop(categorical_cols, axis=1), data_encoded], axis=1)

    # Trim some columns to convert them to numeric values
    data['screen_size'] = data['screen_size'].str.replace(' Inches', '').astype(float)
    data['ram'] = data['ram'].str.replace(' MB', '').str.replace(' GB', '000').astype(float)
    data['price'] = data['price'].str.replace('$', '').str.replace(',', '').astype(float)
    data['harddisk'] = data['harddisk'].str.replace(' MB', '').str.replace(' GB', '000').str.replace(' TB', '000000').astype(float)

    # Handle CPU speed
    if 'cpu_speed' in data.columns:
        data['cpu_speed'] = data['cpu_speed'].apply(_handle_cpu_speed)

    # Convert graphics to 0/1
    data['graphics'] = data['graphics'].map({'Integrated': 0, 'Dedicated': 1})

    # Remove any rows with missing values
    data = data.dropna(axis=0, how='any')

    return data

data = preprocess(data)[['ram', 'harddisk', 'screen_size', 'graphics', 'price', 'brand_Apple']]

# Split the data into features and target
X = data.drop('price', axis=1)
y = data['price']

# Drop rows with any missing values
X = X.dropna(axis=0, how='any')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print(X)

# Train the random forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the MAE
mae = mean_absolute_error(y_test, y_pred)
r2 = model.score(X_test, y_test)
print(f"MAE: ${mae:,.2f}")
print(f"R2: {r2:,.2f}")

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)


# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)