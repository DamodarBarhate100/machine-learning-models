import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler 

scaler_  = StandardScaler()

df = pd.read_csv("data_terrain.csv")

X = df.drop('target', axis=1).to_numpy()
Y = df['target'].to_numpy().reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

def linear_regression(x_train, y_train, lambda_):
    m = len(y_train)
    n = x_train.shape[1]
    iterations = 1000
    learning_rate = 0.01
    theta_j = np.zeros(shape=(n,1))
    theta_0 = 0

    for i in range(iterations):
        y_predicted = x_train @ theta_j  +  theta_0
        errors  = y_predicted - y_train
        gradient_j = (1/m) * x_train.T @ errors + ((lambda_/m) * theta_j)
        gradient_0  = (1/m) * np.sum(errors)

        theta_j = theta_j - learning_rate * gradient_j
        theta_0 = theta_0 - learning_rate * gradient_0

    return theta_j, theta_0

k_folds = 10
X_folds = np.array_split(X_train, k_folds)
Y_folds = np.array_split(Y_train, k_folds)

lambda_ = [0, 0.5, 1, 5, 10, 20, 50]
final_cv_scores = []
for l in lambda_:
    fold_mses = []
    for k in range(k_folds):
        x_validation = X_folds[k]
        y_validation = Y_folds[k]

        x_train_k = np.vstack(X_folds[:k] + X_folds[k+1:])
        y_train_k = np.vstack(Y_folds[:k] + Y_folds[k+1:])

        x_scaled = scaler_.fit_transform(x_train_k)
        x_validation_scaled = scaler_.transform(x_validation)

        m , b = linear_regression(x_scaled, y_train_k, l)

        # Validation
        y_pred = x_validation_scaled @ m + b
        MSE =  mean_squared_error(y_validation, y_pred)
        fold_mses.append(MSE)
        
    avg_cv_mse = np.mean(fold_mses)
    final_cv_scores.append(avg_cv_mse)
    print(f"Lambda {l:<4} | Average Cross Validation MSE: {avg_cv_mse:.2f}")



print("\n Now training and Evaluating on the main train and test dataset:")

X_train_scaled = scaler_.fit_transform(X_train)
X_test_scaled = scaler_.transform(X_test)

m_final , b_final = linear_regression(X_train_scaled, Y_train, 0.5)
print("\n Weights:", m_final)
print("\n Intercept:", b_final)

y_predictions = X_test_scaled @ m_final  + b_final
print("\n Predictions:", y_predictions)

print("MSE:", mean_squared_error(y_pred=y_predictions, y_true=Y_test))



