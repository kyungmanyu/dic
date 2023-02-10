import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('motordata.csv')

X = data.drop('suction', axis=1)
y = data['suction']

print(X)

random_state = 2023
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/10, random_state=random_state)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=2/8,
                                                      random_state=random_state)
max_depths = list(range(1, 10)) + [None]
print(max_depths)

acc_valid = []
f1_valid = []



# for max_depth in max_depths:


#         # 모델 학습
#     model = DecisionTreeClassifier(max_depth=max_depth)
#     model.fit(X_train, y_train)
    
#     # validation 예측
#     y_valid_pred = model.predict(X_valid)
    
#     # 모델 평가 결과 저장
#     acc = accuracy_score(y_valid, y_valid_pred)
#     f1 = f1_score(y_valid, y_valid_pred)
    
#     acc_valid.append(acc)
#     f1_valid.append(f1)


xticks = list(map(str, max_depths))
print(xticks)

tree = DecisionTreeRegressor().fit(X_train, y_train)
tree_prediction = tree.predict(X_test)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("suction Plot")
plt.plot(tree_prediction,color='green')
# plt.plot(y_test,color='r')

plt.show()

# print(tree_prediction)


