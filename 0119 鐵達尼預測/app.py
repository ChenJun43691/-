# import data modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# get data,讀取 Titanic 訓練數據，並顯示前幾筆資料 (df.head()) 以及數據資訊 (df.info())，檢查數據結構、缺失值、資料型別等
df = pd.read_csv(
    "https://raw.githubusercontent.com/ryanchung403/dataset/refs/heads/main/train_data_titanic.csv"
)

df.head()
df.info()

# remove name and ticket columns
df.drop(["Name", "Ticket"], axis=1, inplace=True)
# df = df.drop(["Name", "Ticket"], axis=1)

sns.pairplot(df[['Survived', 'Fare']], dropna=True)
sns.pairplot(df[["Survived", "PassengerId"]], dropna=True)
sns.pairplot(df[["Survived", "Pclass"]], dropna=True)

sns.pairplot(df[["Survived", "Age"]], dropna=True)
sns.pairplot(df[["Survived", "SibSp"]], dropna=True)
sns.pairplot(df[["Survived", "Parch"]], dropna=True)

#print("中文姓名")
#sns.pairplot(df[["Survived", "Sex_num"]], dropna=True)
#sns.pairplot(df[["Survived", "Embarked_num"]], dropna=True)

df.groupby("Survived").mean(numeric_only=True)

df.groupby("Survived").mean(numeric_only=True)

df['SibSp'].value_counts()
df['Parch'].value_counts()
df['Sex'].value_counts()
df['Fare'].value_counts()
df['Embarked'].value_counts()

len(df)/2
df.isnull().sum().sort_values(ascending=False)
df.isnull().sum().sort_values(ascending=False) > len(df)/2

df.drop("Cabin", axis=1, inplace=True)

df.groupby('Sex')['Age'].median().plot(kind='bar')

df.groupby("Sex")["Age"].transform("median")

df['Age'] = df["Age"].fillna(df.groupby('Sex')['Age'].transform('median'))

df["Embarked"].value_counts()
df['Embarked'].value_counts().idxmax()
df["Embarked"] = df["Embarked"].fillna(df['Embarked'].value_counts().idxmax())

df.info()
df['Sex'].value_counts()

df = pd.get_dummies(data=df, dtype=int, columns=['Sex','Embarked'])
df.head()

df.drop("Sex_female", axis=1, inplace=True)

df.corr()

X = df.drop(["Survived", "Pclass"], axis=1)
y = df["Survived"]

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

# train model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)


predictions = lr.predict(X_test)


import joblib

# 儲存訓練好的模型
joblib.dump(lr, "Titanic-LR-20250223.pkl")  # 檔名可自訂

# evaluate model
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

print(accuracy_score(y_test, predictions))
print(recall_score(y_test, predictions))
print(precision_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))

pd.DataFrame(confusion_matrix(y_test, predictions), columns=["預測未存活", "預測存活"], index=["實際上未存活", "實際上存活"])
