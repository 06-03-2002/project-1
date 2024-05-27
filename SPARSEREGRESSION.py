# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:54:52 2024

@author: User
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
import numpy as np

# Read the CSV file
data = pd.read_csv("E:/pupil-mat.csv", encoding='latin1')

# Convert non-numeric columns to categorical type
for column in data.select_dtypes(include=['object']):
    data[column] = data[column].astype(str)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])

# Define columns to encode
columns_to_encode = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

# Initialize LabelEncoder
encoder = LabelEncoder()

# Encode categorical variables
for column in columns_to_encode:
    data[column] = encoder.fit_transform(data[column])

# Define features and target variable
features = data.drop("G1", axis=1)
target = data["G1"]

# Display the transformed dataframe
print(data.head())

# Handle missing values in the features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Handle missing values in the target variable
target_imputed = imputer.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.1)

# Apply polynomial features transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create an instance of the Lasso model
lasso_regressor = Lasso(alpha=0.1)  # You can adjust the alpha parameter for tuning the sparsity

# Fit the model on the polynomial features
lasso_regressor.fit(x_train_poly, y_train)

# Retrieve the coefficients and intercept
coefficients = lasso_regressor.coef_
intercept = lasso_regressor.intercept_

# Retrieve the original feature names
original_feature_names = features.columns

# Generate the polynomial feature names
feature_names = list(original_feature_names)
for feature_idx in poly.powers_:
    if np.sum(feature_idx) > 1:
        feature_name = "*".join(
            [
                f"{name}^{power}"
                for name, power in zip(original_feature_names, feature_idx)
                if power > 0
            ]
        )
        feature_names.append(feature_name)

# Create the equation
equation = "G1 = "
for i, coefficient in enumerate(coefficients):
    if i == 0:
        equation += f"{intercept:.2f}"
    else:
        equation += f" + {coefficient:.2f} * {feature_names[i]}"
print('Coefficients:', coefficients)
print('Intercept:', intercept)
print("Equation:", equation)

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")
# Convert non-numerical values to numerical using LabelEncoder
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])

# Define features and target variable
features = data.drop("G1", axis=1)
target = data["G1"]

# Apply OrdinalEncoder to encode categorical variables
encoder = OrdinalEncoder()
features_encoded = encoder.fit_transform(features)

# Handle missing values in the encoded features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_encoded)

# Handle missing values in the target variable
target_imputed = imputer.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.1)

# Create an instance of the Lasso regression model
lasso = Lasso(alpha=0.1)  # Adjust regularization strength with alpha

# Fit the model on the training data
lasso.fit(x_train, y_train)

# Evaluate the model's accuracy
acc = lasso.score(x_test, y_test)
print("Accuracy:", acc)








import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")
data = pd.read_csv("E:/pupil-mat.csv",encoding='latin1')
data = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

# Convert non-numeric columns to categorical type
for column in data.select_dtypes(include=['object']):
    data[column] = data[column].astype(str)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])
# Encode categorical variables
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])
# Convert non-numerical values to numerical using LabelEncoder
label_encoder = LabelEncoder()
data["school"] = label_encoder.fit_transform(data["school"])
data["sex"] = label_encoder.fit_transform(data["sex"])
data["address"] = label_encoder.fit_transform(data["address"])
data["famsize"] = label_encoder.fit_transform(data["famsize"])
data["Pstatus"] = label_encoder.fit_transform(data["Pstatus"])
data["Mjob"] = label_encoder.fit_transform(data["Mjob"])
data["Fjob"] = label_encoder.fit_transform(data["Fjob"])
data["reason"] = label_encoder.fit_transform(data["reason"])
data["guardian"] = label_encoder.fit_transform(data["guardian"])
data["schoolsup"] = label_encoder.fit_transform(data["schoolsup"])
data["famsup"] = label_encoder.fit_transform(data["famsup"])
data["paid"] = label_encoder.fit_transform(data["paid"])
data["activities"] = label_encoder.fit_transform(data["activities"])
data["nursery"] = label_encoder.fit_transform(data["nursery"])
data["higher"] = label_encoder.fit_transform(data["higher"])
data["internet"] = label_encoder.fit_transform(data["internet"])
data["romantic"] = label_encoder.fit_transform(data["romantic"])
# Define features and target variable
features = data.drop("G1", axis=1)
target = data["G1"]

# Apply OrdinalEncoder to encode categorical variables
encoder = OrdinalEncoder()
features_encoded = encoder.fit_transform(features)

# Handle missing values in the encoded features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_encoded)

# Handle missing values in the target variable
target_imputed = imputer.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.1)

# Create an instance of the Lasso Regression model
model = Lasso(alpha=0.1)  # Adjust regularization strength with alpha

# Fit the model on the training data
model.fit(x_train, y_train)

# Evaluate the model's accuracy on training and testing data
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Read the CSV file and select desired columns

# Label encode categorical variables
label_encoder = LabelEncoder()
for column in data.columns[data.dtypes == object]:
    data[column] = label_encoder.fit_transform(data[column])

# Define features and target variable

# Handle missing values in the features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.1)

# Apply polynomial features transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create an instance of the Lasso model
lasso_regressor = Lasso(alpha=0.1)  # You can adjust the alpha value for desired sparsity

# Fit the model on the polynomial features
lasso_regressor.fit(x_train_poly, y_train)

# Retrieve the coefficients and intercept
coefficients = lasso_regressor.coef_
intercept = lasso_regressor.intercept_

# Retrieve the original feature names
original_feature_names = features.columns

# Generate the polynomial feature names
feature_names = list(original_feature_names)
for feature_idx in poly.powers_:
    if np.sum(feature_idx) > 1:
        feature_name = "*".join(
            [
                f"{name}^{power}"
                for name, power in zip(original_feature_names, feature_idx)
                if power > 0
            ]
        )
        feature_names.append(feature_name)

# Create the equation
equation = "G1 = "
for i, coefficient in enumerate(coefficients):
    if i == 0:
        equation += f"{intercept:.2f}"
    else:
        equation += f" + {coefficient:.2f} * {feature_names[i]}"

print("Equation:", equation)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv", encoding='latin1')
data = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]
# Convert non-numeric columns to categorical type
for column in data.select_dtypes(include=['object']):
    data[column] = data[column].astype(str)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])
# Convert non-numerical values to numerical using LabelEncoder
label_encoder = LabelEncoder()
data["school"] = label_encoder.fit_transform(data["school"])
data["sex"] = label_encoder.fit_transform(data["sex"])
data["address"] = label_encoder.fit_transform(data["address"])
data["famsize"] = label_encoder.fit_transform(data["famsize"])
data["Pstatus"] = label_encoder.fit_transform(data["Pstatus"])
data["Mjob"] = label_encoder.fit_transform(data["Mjob"])
data["Fjob"] = label_encoder.fit_transform(data["Fjob"])
data["reason"] = label_encoder.fit_transform(data["reason"])
data["guardian"] = label_encoder.fit_transform(data["guardian"])
data["schoolsup"] = label_encoder.fit_transform(data["schoolsup"])
data["famsup"] = label_encoder.fit_transform(data["famsup"])
data["paid"] = label_encoder.fit_transform(data["paid"])
data["activities"] = label_encoder.fit_transform(data["activities"])
data["nursery"] = label_encoder.fit_transform(data["nursery"])
data["higher"] = label_encoder.fit_transform(data["higher"])
data["internet"] = label_encoder.fit_transform(data["internet"])
data["romantic"] = label_encoder.fit_transform(data["romantic"])
# Handle missing values in the data
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Separate features and target variable
X = data_imputed[:, :-1]
y = data_imputed[:, -1]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Apply polynomial features transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create an instance of the ElasticNet regression model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust regularization strength with alpha and mix ratio with l1_ratio

# Fit the model on the polynomial features
elastic_net.fit(x_train_poly, y_train)

# Predict 'G1' values for training and testing sets
y_train_pred = elastic_net.predict(x_train_poly)
y_test_pred = elastic_net.predict(x_test_poly)

# Print the predicted 'G1' values
print("Predicted 'G1' values for training set:", y_train_pred)
print("Predicted 'G1' values for test set:", y_test_pred)

# Plot the actual G1 values and the predicted G1 values
import matplotlib.pyplot as plt

plt.scatter(y_test, y_test_pred)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual G1")
plt.ylabel("Predicted G1")
plt.title("Lasso Regression: Actual vs Predicted G1")
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# Separate the features and target variable
X = data_encoded.drop("G1", axis=1)
y = data_encoded["G1"]

# Handle missing values in the data
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1)

# Create an instance of the Lasso regression model
lasso = Lasso(alpha=0.1)  # Adjust regularization strength with alpha

# Fit the model on the training data
lasso.fit(x_train, y_train)

# Get feature names
feature_names = X.columns

# Print the coefficients of the model
print("Coefficients:", lasso.coef_)

# Plot 'G1' against each column
for i, column in enumerate(feature_names):
    plt.scatter(X[column], y)
    plt.xlabel(column)
    plt.ylabel('G1')
    plt.title(f'G1 vs {column}')
    plt.show()

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('E:/pupil-mat.csv')

# Separate the features (input variables) and the target variable
X = data[['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2']]
y = data["G1"] 

# Handle missing values in y
imputer = SimpleImputer(strategy='mean')
y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).flatten()

# Preprocess and encode non-numeric columns in X using LabelEncoder
non_numeric_cols = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
encoder = LabelEncoder()
X_encoded = X.copy()
for col in non_numeric_cols:
    X_encoded[col] = encoder.fit_transform(X[col])

# Handle missing values in X
X_imputed = imputer.fit_transform(X_encoded)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Create an instance of the Lasso Regression model
model = Lasso(alpha=0.1)  # You can adjust the regularization strength by changing alpha

# Fit the model to the scaled data
model.fit(X_scaled, y_imputed)

# Make predictions for the existing data
predictions = model.predict(X_scaled)
print("PREDICTIONS", predictions)
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")
data = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

# Convert non-numerical values to numerical using LabelEncoder
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])

# Define features and target variable
features = data.drop("G1", axis=1)
target = data["G1"]

# Apply OrdinalEncoder to encode categorical variables
encoder = OrdinalEncoder()
features_encoded = encoder.fit_transform(features)

# Handle missing values in the encoded features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features_encoded)

# Handle missing values in the target variable
target_imputed = imputer.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.1)

# Create an instance of the Lasso regression model
lasso = Lasso(alpha=0.1)  # Adjust regularization strength with alpha

# Fit the model on the training data
lasso.fit(x_train, y_train)

# Evaluate the model's accuracy
acc = lasso.score(x_test, y_test)
print("Accuracy:", acc)

# Print the coefficients and intercept
print("Coefficients:", lasso.coef_)
print("Intercept:", lasso.intercept_)
# Print the linear equation
features_names = features.columns
linear_equation = "G1 = "
for i, feature in enumerate(features_names):
    coefficient = lasso.coef_[i]
    linear_equation += f"({coefficient:.2f}) * {feature} + "
linear_equation += f"({lasso.intercept_:.2f})"
print("Linear Equation:", linear_equation)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Read the CSV file and select desired columns

# Convert non-numeric columns to categorical type
for column in data.select_dtypes(include=['object']):
    data[column] = data[column].astype(str)

# Apply OrdinalEncoder to encode categorical variables
encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data)

# Convert non-numerical values to numerical using LabelEncoder
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']):
    data[column] = label_encoder.fit_transform(data[column])

# Define features and target variable
features = pd.DataFrame(data_encoded, columns=data.columns).drop("G1", axis=1)
target = data["G1"]

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.1)

# Create an instance of the Logistic Regression model
logistic = LogisticRegression()

# Fit the model on the training data
logistic.fit(x_train, y_train)

# Predict the target variable on the test data
y_pred = logistic.predict(x_test)

# Evaluate the model's accuracy
accuracy = logistic.score(x_test, y_test)
print("Test Accuracy:", accuracy)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")
data = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

# Convert non-numerical values to numerical using LabelEncoder
label_encoder = LabelEncoder()
data["school"] = label_encoder.fit_transform(data["school"])
data["sex"] = label_encoder.fit_transform(data["sex"])
data["address"] = label_encoder.fit_transform(data["address"])
data["famsize"] = label_encoder.fit_transform(data["famsize"])
data["Pstatus"] = label_encoder.fit_transform(data["Pstatus"])
data["Mjob"] = label_encoder.fit_transform(data["Mjob"])
data["Fjob"] = label_encoder.fit_transform(data["Fjob"])
data["reason"] = label_encoder.fit_transform(data["reason"])
data["guardian"] = label_encoder.fit_transform(data["guardian"])
data["schoolsup"] = label_encoder.fit_transform(data["schoolsup"])
data["famsup"] = label_encoder.fit_transform(data["famsup"])
data["paid"] = label_encoder.fit_transform(data["paid"])
data["activities"] = label_encoder.fit_transform(data["activities"])
data["nursery"] = label_encoder.fit_transform(data["nursery"])
data["higher"] = label_encoder.fit_transform(data["higher"])
data["internet"] = label_encoder.fit_transform(data["internet"])
data["romantic"] = label_encoder.fit_transform(data["romantic"])
# ... (continue encoding other columns)

# Define features and target variable
features = data.drop("sex", axis=1)
target = data["sex"]

# Apply OrdinalEncoder to encode categorical variables
encoder = OrdinalEncoder()
features_encoded = encoder.fit_transform(features)

# Handle missing values in the encoded features
imputer = SimpleImputer(strategy='most_frequent')
features_imputed = imputer.fit_transform(features_encoded)

# Handle missing values in the target variable
target_imputed = imputer.fit_transform(target.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target_imputed, test_size=0.1)

# Create an instance of the ElasticNet regression model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust regularization strength with alpha and mix ratio with l1_ratio

# Fit the model on the training data
elastic_net.fit(x_train, y_train)

# Evaluate the model's accuracy
acc = elastic_net.score(x_test, y_test)
print("Accuracy:", acc)

# Print the coefficients and intercept
print("Coefficients:", elastic_net.coef_)
print("Intercept:", elastic_net.intercept_)

# Print the linear equation
features_names = features.columns
linear_equation = "sex ="
for i, feature in enumerate(features_names):
    coefficient = elastic_net.coef_[i]
    linear_equation += f"({coefficient:.2f}) * {feature} + "
linear_equation += f"({elastic_net.intercept_:.2f})"
print("Linear Equation:", linear_equation)
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('E:/pupil-mat.csv')

# Separate the features (input variables) and the target variable
X = data[['school', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]
y = data["sex"]

# Preprocess and encode all columns using OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Handle missing values in X
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X_encoded)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Encode the target variable y
encoder_y = OrdinalEncoder()
y_encoded = encoder_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Create an instance of the Elastic Net model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust regularization strength with alpha and mix ratio with l1_ratio

# Fit the model to the scaled data
elastic_net.fit(X_scaled, y_encoded)

# Make predictions for the existing data
predictions = elastic_net.predict(X_scaled)
print("PREDICTIONS:", predictions)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")
data = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

# Convert non-numerical values to numerical using LabelEncoder
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Define features and target variable
X = data.drop(columns=['sex'])
y = data['sex']

# Handle missing values in the data
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1)

# Create an instance of the ElasticNet model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust regularization strength with alpha and mix ratio with l1_ratio

# Fit the model on the training data
elastic_net.fit(x_train, y_train)

# Evaluate the model's accuracy on the training data
train_acc = elastic_net.score(x_train, y_train)
print("Training Accuracy:", train_acc)

# Evaluate the model's accuracy on the test data
test_acc = elastic_net.score(x_test, y_test)
print("Test Accuracy:", test_acc)
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")
data = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

# Encode categorical variables
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# Define features and target variable
features = data_encoded.drop("sex", axis=1)
target = data_encoded["sex"]

# Handle missing values in the features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.1)

# Apply polynomial features transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create an instance of the ElasticNet model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

# Fit the model on the polynomial features
elastic_net.fit(x_train_poly, y_train)

# Retrieve the coefficients and intercept
coefficients = elastic_net.coef_
intercept = elastic_net.intercept_

# Retrieve the original feature names
original_feature_names = features.columns

# Generate the polynomial feature names
feature_names = list(original_feature_names)
for feature_idx in poly.powers_:
    if np.sum(feature_idx) > 1:
        feature_name = "*".join(
            [
                f"{name}^{power}"
                for name, power in zip(original_feature_names, feature_idx)
                if power > 0
            ]
        )
        feature_names.append(feature_name)

# Create the equation
equation = "sex = "
for i, coefficient in enumerate(coefficients):
    if i == 0:
        equation += f"{intercept:.2f}"
    else:
        equation += f" + {coefficient:.2f} * {feature_names[i]}"
        
print("Equation:", equation)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet

# Read the CSV file
data = pd.read_csv("E:/pupil-mat.csv")

# Select relevant columns
data = data[['school', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'sex']]

# Encode categorical variables
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_encoded)

# Split features and target variable
X = data_imputed[:, :-1]
y = data_imputed[:, -1]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of ElasticNet model
# You can adjust the parameters alpha (regularization strength) and l1_ratio (L1 ratio) as needed
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train the model
elastic_net.fit(x_train, y_train)

# Make predictions for training and testing sets
y_train_pred = elastic_net.predict(x_train)
y_test_pred = elastic_net.predict(x_test)

# Print the predicted values for training and testing sets
print("Predicted 'sex' values for training set:", y_train_pred)
print("Predicted 'sex' values for test set:", y_test_pred)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("E:/pupil-mat.csv")

# Select relevant columns
data = data[['school', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'sex']]

# Encode categorical variables
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_encoded)

# Split features and target variable
X = data_imputed[:, :-1]
y = data_imputed[:, -1]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of ElasticNet model
# You can adjust the parameters alpha (regularization strength) and l1_ratio (L1 ratio) as needed
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train the model
elastic_net.fit(x_train, y_train)

# Make predictions for training and testing sets
y_train_pred = elastic_net.predict(x_train)
y_test_pred = elastic_net.predict(x_test)

# Plot the actual sex values and the predicted sex values
plt.scatter(y_test, y_test_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual sex")
plt.ylabel("Predicted sex")
plt.title("Elastic Net Regression: Actual vs Predicted sex")
plt.show()

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as plt

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")
data = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

# Encode categorical variables
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# Define features and target variable
features = data_encoded.drop("sex", axis=1)
target = data_encoded["sex"]

# Handle missing values in the features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.1)

# Apply polynomial features transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Create an instance of the ElasticNet model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Adjust regularization strength with alpha and mix ratio with l1_ratio

# Fit the model on the polynomial features
elastic_net.fit(x_train_poly, y_train)

# Predict 'sex' values using the trained model
y_pred = elastic_net.predict(x_test_poly)

# Create scatter plots between predicted sex and other columns
for col_idx, col_name in enumerate(features.columns):
    plt.scatter(x_test_poly[:, col_idx], y_pred, label="Predicted sex")
    plt.xlabel(col_name)
    plt.ylabel("Predicted sex")
    plt.title(f"Predicted sex vs {col_name}")
    plt.legend()
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Read the CSV file and select desired columns
data = pd.read_csv("E:/pupil-mat.csv")
data = data[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']]

# Encode categorical variables
label_encoder = LabelEncoder()
data_encoded = data.apply(label_encoder.fit_transform)

# Define features and target variable
features = data_encoded.drop("sex", axis=1)
target = data_encoded["sex"]

# Handle missing values in the features
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.1)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create an instance of the LogisticRegression model with elastic net penalty
classifier = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)  # Adjust the l1_ratio as needed

# Fit the model
classifier.fit(x_train_scaled, y_train)

# Predict 'sex' values using the trained model
y_pred = classifier.predict(features_imputed)

# Retrieve the column names
column_names = features.columns

# Plot predicted 'school' against each column
for column in column_names:
    unique_values = data[column].unique()
    num_unique = len(unique_values)
    plt.figure(figsize=(12, 6))
    for i, value in enumerate(unique_values):
        plt.subplot(1, num_unique, i+1)
        plt.bar([0, 1], [np.sum((features[column] == value) & (y_pred == 0)),
                         np.sum((features[column] == value) & (y_pred == 1))], color=['blue', 'red'])
        plt.xlabel("sex")
        plt.ylabel("Count")
        plt.title(f"{column} = {value}")
        plt.xticks([0, 1], label_encoder.inverse_transform([0, 1]))  # Use inverse_transform to get the original labels
    plt.tight_layout()
    plt.show()
