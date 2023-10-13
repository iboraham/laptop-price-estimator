# Laptop Price Estimation Project 💻💰

This project aims to estimate laptop prices using a dataset of laptops collected from Amazon.com. The dataset contains product details such as rating, price, operating system, title, review count, and display size, among others. The dataset consists of approximately 4.5K data points, with 14 distinct features.

## Dataset Description 📊

The dataset consists of the following columns:

- **brand**: Make of laptop
- **model**: Model of laptop
- **screen size**: The size of the laptop's display in inches.
- **color**: Color of laptop
- **hard disk**: Hard disk size in GBs or TBs
- **CPU**: Processor installed
- **RAM**: RAM size installed in the laptop
- **Operating system (OS)**: The operating system of the laptop.
- **Special Features**: Extra features
- **Graphics card**: Graphics card
- **Graphics coprocessor**: Graphics card
- **CPU rating**: CPU rating
- **Rating**: The average customer rating out of 5 as of October 2023.
- **Price**: The price of the product in USD as of October 2023.

## Data Preprocessing 🛠️

The `train.py` script is used to preprocess the data and train a machine learning model for price estimation. Here's a brief overview of the data preprocessing steps:

1. **Loading Data**: The dataset is loaded from a CSV file named `data.csv`.

2. **Data Preprocessing Functions**: The script defines several functions for data preprocessing, such as `_handle_cpu_speed` to handle CPU speed values.

3. **Data Preprocessing**: The `preprocess` function is used for data preprocessing. It performs the following operations:
   - Removing duplicate rows
   - Dropping columns with too many missing values
   - Converting categorical columns to one-hot encoding
   - Trimming and converting some columns to numeric values (e.g., screen size, RAM, price)
   - Handling CPU speed values
   - Converting the 'graphics' column to binary (0/1)
   - Removing rows with missing values

4. **Feature Selection**: The script selects specific columns for modeling, including 'ram', 'harddisk', 'screen_size', 'graphics', 'price', and one-hot encoded 'brand' (e.g., 'brand_Apple').

5. **Splitting Data**: The data is split into features (X) and the target (y) for training a machine learning model. Rows with missing values are dropped from the features.

6. **Random Forest Model**: A Random Forest Regressor model is used for price estimation.

7. **Model Training and Evaluation**: The model is trained on the training data, and predictions are made on the test data. Mean Absolute Error (MAE) and R-squared (R2) are calculated to evaluate the model's performance.

8. **Feature Importance**: The script also calculates feature importances, which can provide insights into which features are most influential in predicting laptop prices.

9. **Model Saving**: The trained model is saved to a file named `model.pkl`.

## Usage 🚀

To use this project, follow these steps:

1. Download the dataset from Kaggle using the following command:
   ```
   kaggle datasets download -d talhabarkaatahmad/laptop-prices-dataset-october-2023
   ```

2. Rename the downloaded dataset to `data.csv`.

3. Run the `train.py` script to preprocess the data, train the model, and save it.
<div style="background-color: #f2f2f2; padding: 20px; text-align: center;">
    <h2>Model Evaluation Scores</h2>
    <div style="display: flex; justify-content: space-around; margin-top: 20px;">
        <div style="background-color: #4CAF50; color: white; padding: 15px; border-radius: 5px;">
            <h3>MAE</h3>
            <p style="font-size: 24px; margin: 0;">$384.23</p>
        </div>
        <div style="background-color: #3498db; color: white; padding: 15px; border-radius: 5px;">
            <h3>R2</h3>
            <p style="font-size: 24px; margin: 0;">0.60</p>
        </div>
    </div>
</div>

4. Use the trained model to estimate laptop prices.

For more information about the dataset, you can refer to the [Kaggle dataset page](https://www.kaggle.com/talhabarkaatahmad/laptop-prices-dataset-october-2023).

Feel free to explore the code and adapt it to your specific requirements for laptop price estimation. 🤖📈