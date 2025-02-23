# import model
import joblib
model_pretrained = joblib.load("Titanic-LR-20250223.pkl")

import pandas as pd
df_test = pd.read_csv("test.csv")
df_test.info()

df_test.drop(["Name", "Ticket"], axis=1, inplace=True)
df_test.drop("Cabin", axis=1, inplace=True)

df_test["Age"] = df_test["Age"].fillna(df_test.groupby("Sex")["Age"].transform("median"))

df_test["Fare"].value_counts().idxmax()
df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].value_counts().idxmax())

df_test = pd.get_dummies(data=df_test, dtype=int, columns=["Sex", "Embarked"])

df_test.drop("Sex_female", axis=1, inplace=True)
df_test.drop("Pclass", axis=1, inplace=True)

df_test.info()
# predict
predictions2 = model_pretrained.predict(df_test)

#save to csv
forSubmissionDF = pd.DataFrame(
    {"PassengerId": df_test["PassengerId"], 
     "Survived": predictions2})
forSubmissionDF.to_csv("for_submission_20250223.csv", index=False)