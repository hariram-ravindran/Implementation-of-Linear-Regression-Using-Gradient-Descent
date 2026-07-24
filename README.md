## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed. 

2.Scale the features using a standard scaler to normalize the data. 

3.Initialize model parameters (theta) and add an intercept term to the feature set. 

4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function. 

5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:HARIRAM R
RegisterNumber:212224240050 
*/
```
```

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def linear_regression(X, y, iters=1000, learning_rate=0.01):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept term
    theta = np.zeros((X.shape[1], 1))

    for _ in range(iters):
        predictions = X.dot(theta)
        errors = predictions - y.reshape(-1, 1)
        gradient = (1 / X.shape[0]) * X.T.dot(errors)
        theta -= learning_rate * gradient

    return theta

data = pd.read_csv('exp_3_50_Startups.csv', header=0)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [3])
], remainder='passthrough')

X = ct.fit_transform(X)

y = y.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

theta = linear_regression(X_scaled, y, iters=1000, learning_rate=0.01)

new_data = np.array([165349.2, 136897.8, 471784.1, 'New York']).reshape(1, -1)  # Example new data
new_data_scaled = scaler.transform(ct.transform(new_data))

new_prediction = np.dot(np.append(1, new_data_scaled), theta)

print(f"Predicted value: {new_prediction[0]}")
data.head()
```

## Output:
<img width="621" height="268" alt="Screenshot 2026-07-24 091535" src="https://github.com/user-attachments/assets/06aca2a8-883b-4050-867d-2cb2fcd4e257" />
<img width="818" height="276" alt="Screenshot 2026-07-24 091540" src="https://github.com/user-attachments/assets/06419fa1-8c8f-4702-8804-2e3dbd4cbfec" />
<img width="277" height="60" alt="Screenshot 2026-07-24 091546" src="https://github.com/user-attachments/assets/f680d495-88eb-4c60-8bd4-40f9bd5c3baf" />
<img width="406" height="40" alt="Screenshot 2026-07-24 091550" src="https://github.com/user-attachments/assets/8dd2f328-2faf-4614-97ed-2d13898ff290" />
<img width="622" height="331" alt="Screenshot 2026-07-24 091556" src="https://github.com/user-attachments/assets/f31befb0-6d65-436d-9ff3-5bf3a50801db" />
<img width="817" height="486" alt="Screenshot 2026-07-24 091600" src="https://github.com/user-attachments/assets/6fb8a329-7607-44a5-8d76-1fbdced15ffc" />
<img width="492" height="36" alt="Screenshot 2026-07-24 091608" src="https://github.com/user-attachments/assets/bba54048-69af-4387-87ca-286f97e125ba" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
