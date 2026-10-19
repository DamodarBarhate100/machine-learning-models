import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,  confusion_matrix


df = pd.read_csv("01_Datasets/bank_loan_defaults.csv")

X = df.drop(labels='Default', axis=1).to_numpy()
y = df['Default'].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=100,splitter='best')

tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
print("Predicitons:", predictions)


print("\n Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_true=y_test, y_pred=predictions))
print("Precision:", precision_score(y_true=y_test, y_pred=predictions))
print("Recall:", recall_score(y_true=y_test, y_pred=predictions))
print("Confusion Matrix:\n", confusion_matrix(y_true=y_test, y_pred=predictions))


feature_names = list(df.columns[:-1]) 
target_name = list(df.iloc[:, -1].unique().astype(str)) 

plt.figure(figsize=(25, 20)) 
plot_tree(decision_tree=tree, rounded=True, fontsize=10, feature_names=feature_names, class_names=target_name, filled=True)

plt.savefig("03_sklearn Implementations/Images/decision_tree_splits.png")