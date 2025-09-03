# Predicting Vehicle Fuel Efficiency (MPG) with Neural Networks
Predicting Vehicle Fuel Efficiency (MPG) with a Neural Network

## Project Overview
The goal of this project is to answer a classic automotive question: What makes a car fuel-efficient? To do this, we'll build a deep learning model that predicts a car's fuel consumption (in miles-per-gallon) based on its specifications, such as the number of cylinders, horsepower, and weight.

## The Dataset

We use the classic "Auto MPG" dataset from Seaborn library. The dataset contains 398 rows and 9 columns. The columns are as follows:
- `mpg`: Miles Per Gallon (target variable)
- `cylinders`: Number of cylinders
- `displacement`: Engine displacement
- `horsepower`: Engine horsepower
- `weight`: Vehicle weight
- `acceleration`: Time to accelerate from 0 to 60 mph
- `model_year`: Model year of the vehicle
- `origin`: Region of origin (USA, Europe, Japan)
- `name`: Vehicle name (unique identifier, not used for prediction)


## 1. Exploratory Data Analysis (EDA)
The first step is to understand the data. EDA involves loading the dataset, inspecting its structure, identifying potential issues like missing values, and visualizing the relationships between different features.
Key Findings:
- Dataset Shape: The dataset contains 398 observations and 9 features.
- Features: mpg, cylinders, displacement, horsepower, weight, acceleration, model_year, origin, and name.
- Target Variable: mpg is our target variable, which we want to predict.
- Missing Values: The horsepower column contains a few missing values that need to be handled.
- Correlations: We observed strong negative correlations between mpg and features like cylinders, displacement, and weight. This makes intuitive sense—heavier cars with more cylinders tend to have lower fuel efficiency. acceleration and model_year show a positive correlation with mpg.


## 2. Data Preparation and Preprocessing
This phase involves cleaning the data and transforming it into a suitable format for our neural network.
Steps Taken:
- Handling Missing Data: The missing **horsepower** values were identified. We used imputation (filling with the median).
- Handling Categorical Data: The **origin** column is categorical (USA, Europe, Japan). Since machine learning models work with numbers, we converted this feature into a numerical format using one-hot encoding. This creates new binary columns (USA, Europe, Japan) for each origin.
- Splitting the Data: The dataset was split into a training set (80%) and a testing set (20%). The model learns from the training data, and its performance is evaluated on the unseen testing data.
- Feature Normalization: The features in the dataset have very different scales and ranges (e.g., weight is in thousands, while cylinders is a single digit). Normalization (specifically, standardization) scales all features to have a mean of 0 and a standard deviation of 1. This is a critical step that helps the neural network converge faster and learn more effectively.

## 3. Building the Neural Network
With the data prepared, we can now define our model architecture using the Keras API within TensorFlow.

**Model Architecture:** 
We constructed a Sequential model with the following layers:
- Input Layer: A Dense layer with 64 neurons and a ReLU (Rectified Linear Unit) activation function. The shape of the input is determined by the number of features in our prepared data.
- Hidden Layer: A second Dense layer, also with 64 neurons and ReLU activation. This layer helps the model learn more complex patterns in the data.
- Output Layer: A final Dense layer with a single neuron. This neuron outputs our prediction—a single continuous value for mpg. It has no activation function (or a linear activation), which is standard for regression problems.

**Compilation:** 
The model was compiled with:
- Loss Function: mean_squared_error. This measures how far, on average, the model's predictions are from the actual values and penalizes larger errors more heavily.
- Optimizer: Adam. An efficient and widely-used optimization algorithm. We also monitored mean_squared_error for additional insight.

## 4. Training and Evaluation
The model was trained on the normalized training data for 100 epochs. An Early Stopping callback was used to monitor the validation loss. This technique automatically stops the training process if the model's performance on a validation set stops improving, which is a great way to prevent overfitting.

Results:The model performed well on the unseen test data.Test Mean Absolute Error (MAE): The final model achieved an MAE of approximately 2.5 on the test set. This means, on average, the model's MPG predictions are off by only 2.5 miles per gallon. We visualized the model's predictions against the true values, which showed a strong linear relationship, indicating accurate predictions. 

## 5. How to Run This Project
Clone the repository.
Install the required libraries:pip install tensorflow pandas numpy seaborn matplotlib scikit-learn
Execute the Python script:python mpg_prediction_nn.py
The script will process the data, train the model, evaluate it, and display the resulting plots.