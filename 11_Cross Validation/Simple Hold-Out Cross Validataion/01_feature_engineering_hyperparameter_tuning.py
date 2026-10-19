import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("hypertuning_data.csv")

X = df["x"].to_numpy().reshape(-1, 1)
Y = df["target"].to_numpy().reshape(-1, 1)

x_train, x_temp, y_train, y_temp = train_test_split(X,Y, train_size=0.6, random_state=42)
x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()

def linear_regression(x_, y_, lambda_=0):
    m = len(y_)
    n = x_.shape[1]
    iterations = 5000
    learning_rate = 0.01
    theta_j = np.zeros(shape=(n,1))
    theta_0 = 0
    for i in range(iterations):
        y_predicted =   x_ @ theta_j + theta_0
        errors = y_predicted - y_

        gradient_j = (1/m) * x_.T @ errors +  ((lambda_/m) * theta_j)
        gradient_0 = (1/m) * np.sum(errors)
        theta_j = theta_j - learning_rate * gradient_j
        theta_0 = theta_0 - learning_rate * gradient_0
    return theta_j, theta_0        



results = [] 
def predict(x_t, y_t, m, b, i, lamda):
    y_pred = x_t @ m  + b
    mse_val = mean_squared_error(y_pred=y_pred, y_true=y_t)
    results.append((mse_val, i, lamda)) 
    print(f"Degree {i:<2} | Lambda {lamda:<4} | Dev MSE: {mse_val:.2f}")

lambda_ = [0, 0.001, 0.01, 0.1, 1, 10, 50, 100, 1000]
for i in range(1, 11):
    x_dev_poly = np.column_stack(([x_dev **j for j in range(1, i+1)]))
    x_train_poly = np.column_stack(([x_train ** j for j in range(1 , i + 1)]))
    x_train_scaled = scaler.fit_transform(x_train_poly)
    x_dev_scaled  = scaler.transform(x_dev_poly)

    for j in lambda_:
        m, b = linear_regression(x_train_scaled, y_train, lambda_= j)
        predict(x_dev_scaled, y_dev, m, b , i , j )



print("\n" + "="*50)
best_run = min(results, key=lambda item: item[0])
best_mse, best_degree, best_lambda = best_run

print(f"Optimal Polynomial Degree: {best_degree}")
print(f"Optimal Lambda (L2):      {best_lambda}")
print(f"Lowest Dev MSE:           {best_mse:.2f}")


