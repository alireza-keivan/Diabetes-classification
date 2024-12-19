# Diabetes-classification
Certainly! Here's the GitHub README in the specified format:

---

# Diabetes Classification Using K-Nearest Neighbors (KNN)

This repository contains one of my first codes for classification using **K-Nearest Neighbors (KNN)** on the **Pima Indians Diabetes** dataset. The project covers data cleaning, visualization, model training, and evaluation. Comments are added using **Gemini**.

## Table of Contents
1. [Description](#description)
2. [Libraries and Setup](#libraries-and-setup)
3. [Data Loading and Inspection](#data-loading-and-inspection)
4. [Handling Missing Values](#handling-missing-values)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Data Preprocessing](#data-preprocessing)
7. [Train-Test Split](#train-test-split)
8. [Model Training and Evaluation](#model-training-and-evaluation)
9. [Confusion Matrix](#confusion-matrix)

---

## Description
This project demonstrates a classification task using **K-Nearest Neighbors (KNN)** on the **Pima Indians Diabetes** dataset. It covers all essential steps, from data loading and cleaning to model training and evaluation. The goal is to predict whether a person has diabetes based on various medical features. This was one of my first codes for classification, and the comments have been added using **Gemini**.

---

## Libraries and Setup
#### `Libraries required and setup`
- **Purpose**: Import necessary libraries to handle data, visualize it, and create machine learning models.
- **Steps**:
  1. Import libraries such as `NumPy`, `Pandas`, and `Seaborn` for data manipulation and visualization.
  2. Suppress warnings for cleaner outputs.

```python
from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
```

---

## Data Loading and Inspection
#### `Loading and inspecting the dataset`
- **Input**: Path to the CSV dataset.
- **Output**: Initial overview of the dataset, including its structure and descriptive statistics.
- **Purpose**: Load the dataset and understand its structure.
- **Steps**:
  1. Load the dataset with `pd.read_csv()`.
  2. Display the first few rows using `head()`.
  3. Inspect the data types and missing values with `info()` and `describe()`.

```python
df = pd.read_csv(r'C:\Users\win10\Downloads\diabetes.csv')
df.head()

df.info(verbose=True)
df.describe()
```

---

## Handling Missing Values
#### `Replacing invalid values with NaN and handling missing data`
- **Input**: Original dataframe.
- **Output**: Dataframe with missing values filled using column means.
- **Purpose**: Replace inappropriate zero values with `NaN` and fill the missing values.
- **Steps**:
  1. Replace `0` values in specific columns with `NaN` since these values are invalid for attributes like `Glucose` and `BloodPressure`.
  2. Use `mean()` to fill the missing values for numeric columns.
  3. Visualize missing values using the `missingno` library.

```python
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(df_copy.isnull().sum())

import missingno as msno
p = msno.bar(df_copy)

df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].mean(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].mean(), inplace=True)
```

---

## Exploratory Data Analysis (EDA)
#### `Visualizing data distributions and correlations`
- **Input**: Cleaned dataframe.
- **Output**: Histograms of key features and a correlation matrix.
- **Purpose**: Understand the distribution of the data and relationships between features.
- **Steps**:
  1. Plot histograms to visualize the distribution of features like `Glucose`, `BMI`, and others.
  2. Use a heatmap to show correlations between features.

```python
p = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].hist(figsize=(20,20))

sns.countplot(y=df.dtypes, data=df)
plt.xlabel('count of each data type')
plt.ylabel('data types')
plt.show()

plt.figure(figsize=(12,10))
p = sns.heatmap(df_copy.corr(), annot=True, cmap='RdYlGn')
```

---

## Data Preprocessing
#### `Feature scaling using StandardScaler`
- **Input**: Cleaned dataframe.
- **Output**: Scaled features in a new dataframe `X` and the target variable `y`.
- **Purpose**: Standardize features so they have zero mean and unit variance.
- **Steps**:
  1. Apply `StandardScaler` to the features excluding the target (`Outcome`).
  2. Store the transformed features and the target variable separately.

```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X = pd.DataFrame(sc_X.fit_transform(df_copy.drop(["Outcome"], axis=1)),
                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = df_copy.Outcome
```

---

## Train-Test Split
#### `Splitting the data into training and testing sets`
- **Input**: Scaled features and target variable.
- **Output**: Training and testing datasets.
- **Purpose**: Split the dataset into training and testing sets for model validation.
- **Steps**:
  1. Use `train_test_split()` with an 85/15 split, ensuring stratified sampling to maintain class distribution.

```python
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=20, stratify=y)
```

---

## Model Training and Evaluation
#### `Training the KNN model and tuning hyperparameters`
- **Input**: Training dataset.
- **Output**: Performance metrics for various values of K (neighbors).
- **Purpose**: Train and evaluate KNN models with different K values to select the best one.
- **Steps**:
  1. Train a series of KNN models with K values ranging from 1 to 15.
  2. Record the accuracy for both training and testing datasets.
  3. Plot the results to visualize the best-performing K.

```python
from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1, 16):
    knn = KNeighborsClassifier(i)
    knn.fit(train_x, train_y)
    
    train_scores.append(knn.score(train_x, train_y))
    test_scores.append(knn.score(test_x, test_y))

plt.figure(figsize=(12,10))
p = sns.lineplot(range(1, 16), train_scores, markers='*', label='Train score')
p = sns.lineplot(range(1, 16), test_scores, markers='o', label='Test score')
```

---

## Confusion Matrix
#### `Evaluating model performance with a confusion matrix`
- **Input**: Test dataset.
- **Output**: Confusion matrix showing the number of correct and incorrect predictions.
- **Purpose**: Visualize the performance of the KNN model on the test dataset.
- **Steps**:
  1. Train the KNN model with the best K value.
  2. Predict the outcomes on the test set and generate a confusion matrix.
  3. Plot the confusion matrix using a heatmap.

```python
from sklearn.metrics import confusion_matrix

# Train the best KNN model
knn = KNeighborsClassifier(10)
knn.fit(train_x, train_y)

# Predict on the test set
y_pred = knn.predict(test_x)

# Confusion matrix
cnf_matrix = confusion_matrix(test_y, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
```

---

## Conclusion
This project demonstrates the complete machine learning pipeline for classification using **K-Nearest Neighbors (KNN)**, from data loading and preprocessing to model evaluation. Feel free to explore and modify the code for further experimentation!

---
