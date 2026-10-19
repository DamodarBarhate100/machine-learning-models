import pandas as pd
from sklearn.model_selection import train_test_split

# Class 
from DecisionTree import *

df = pd.read_csv("01_Datasets/bank_loan_defaults.csv")

X  = df.drop(columns='Default').to_numpy()
y = df['Default'].to_numpy()

X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=45)

dt = DecisionTree(min_samples_split=2, max_depth=100)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
print(predictions)


accuracy = np.mean(predictions == y_test) * 100
print("Accuracy:", accuracy)



