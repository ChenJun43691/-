鐵達尼生存預測
from sklearn import preprocessing 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

	•	pandas：幫助我們讀取 CSV 檔案。
	•	numpy：處理數學運算（像是平均值）。
	•	seaborn、matplotlib：畫圖表來觀察數據。
	•	sklearn：這是 機器學習工具箱，我們用它來訓練模型（隨機森林 RandomForestClassifier）。
 train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv('gender_submission.csv')

train.info()


	•	讀取 訓練資料 (train.csv)，這裡面有 乘客的資訊 和 是否生還 (Survived)。
	•	讀取 測試資料 (test.csv)，這裡面 沒有 Survived，我們要預測！
	•	讀取 提交格式 (gender_submission.csv)，我們預測完 要按照這個格式存檔。

 
 data = pd.concat([train, test], ignore_index=True, sort=False)
data.reset_index(inplace=True, drop=True)



	•	這裡 畫圖表來觀察數據：
	•	Pclass（艙等）跟生還率的關係
	•	Sex（性別）跟生還率的關係
	•	我們可以發現：
	•	頭等艙 (Pclass=1) 生還率較高
	•	女性 (Sex=Female) 生還率比較高


 data['Embarked'] = data['Embarked'].fillna('S')
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

填補空值：
	•	Embarked（登船地點）有些人沒填，我們補上 最多人登船的地點 S。
	•	Fare（票價）有缺失值，我們用 平均票價來填補。

 data['Title1'] = data['Name'].str.split(", ", expand=True)[1]
data['Title1'] = data['Title1'].str.split(".", expand=True)[0]


	•	有些票號 (Ticket) 有特別的開頭字母，我們取這個字母來當作類別資訊。

 data["Cabin"] = data['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin')

	•	Cabin（房間號碼）有時候會缺失，我們只取它的 第一個字母。

 data['Family_Size'] = data['Parch'] + data['SibSp']


	•	我們新增 家庭人數特徵，因為家庭較大可能影響生還率。

 
 for col in ['Sex', 'Embarked', 'Pclass', 'Title1', 'Title2', 'Cabin', 'Ticket_info']:
    data[col] = data[col].astype('category').cat.codes

	•	Sex、Embarked 等 文字類別變數 需要變成 數字，讓機器學習模型能使用。

 rfModel_age = RandomForestRegressor(n_estimators=2000, random_state=42)
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2', 'Cabin', 'Ticket_info']
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

ageNullValues = rfModel_age.predict(dataAgeNull[ageColumns])
dataAgeNull.loc[:, "Age"] = ageNullValues

	•	Age（年齡）有缺失值，我們用 隨機森林 (RandomForestRegressor) 預測年齡。

 rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)


	•	訓練 隨機森林 (RandomForestClassifier) 來預測生還機率。
	•	oob_score_ 讓我們可以知道模型的表現。


 rf_res = rf.predict(dataTest)
submit['Survived'] = rf_res.astype(int)
submit.to_csv('submit.csv', index=False)


	•	預測測試集 (test.csv) 的存活情況。
	•	儲存結果 以便上傳到 Kaggle。
