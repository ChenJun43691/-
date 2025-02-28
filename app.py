from sklearn import preprocessing 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
# sklearn：用來處理 機器學習相關的功能，像是資料預處理（preprocessing）、
# 模型選擇（GridSearchCV）、隨機森林（RandomForestClassifier、RandomForestRegressor）。
# import data modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #用來 視覺化資料，例如畫出 countplot、histplot。
import seaborn as sns

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv('gender_submission.csv')

train.head(5)
train.info()

# 正確的合併方式 合併 訓練集與測試集，方便做 特徵工程。
data = pd.concat([train, test], ignore_index=True, sort=False)

# 重設索引 重設索引，避免舊的索引影響後續操作。
data.reset_index(inplace=True, drop=True)

# 檢查合併後的 DataFrame
data.info()

#資料視覺化
sns.countplot(x="Pclass", hue="Survived", data=data) #顯示 不同艙等的生還率。
plt.show()

sns.countplot(x="Sex", hue="Survived", data=data) #男女生還率的差異。

g = sns.FacetGrid(data, col='Survived')



sns.histplot(data["Age"], kde=True)  # `kde=True` 會顯示密度曲線
sns.displot(data, x="Age", kde=True)  # `data` 是整個 DataFrame
sns.countplot(data=data, x='Embarked', hue='Survived')  # 不同登船港口的生還率。

plt.show()

#填補缺失值
data['Embarked'] = data['Embarked'].fillna('S') #Embarked 缺失值補 S（最多人從 S 登船）
data['Fare'] = data['Fare'].fillna(data['Fare'].mean()) #Fare 缺失值補平均數。

#提取標題（Title）特徵
data['Title1'] = data['Name'].str.split(", ", expand=True)[1]
data['Title1'] = data['Title1'].str.split(".", expand=True)[0]

data['Title1'].unique()

data['Title2'] = data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
         ['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr','Mrs'])
data['Title2'].unique()

#提取票務資訊 清理 Ticket，保留 開頭字母 來代表票務類別（例如 PC 12345 → PC）。
data['Ticket_info'] = data['Ticket'].apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')

data['Ticket_info'].unique()

#清理 Cabin 取 Cabin 的第一個字母（A, B, C…）。缺失值標記為 NoCabin。
data["Cabin"] = data['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin')
data["Cabin"].unique()

g = sns.FacetGrid(data, col='Survived')
g.map(sns.histplot, 'Parch', kde=False)  # 替換 distplot 為 histplot

g = sns.FacetGrid(data, col='Survived')
g.map(sns.histplot, 'SibSp', kde=False)  # 替換 distplot 為 histplot


# 創建 Family_Size 特徵 家庭大小 = Parch（父母/小孩數）+ SibSp（兄弟姐妹/配偶數）
data['Family_Size'] = data['Parch'] + data['SibSp']

# 替換 distplot，改用 histplot
g = sns.FacetGrid(data, col='Survived')
g.map(sns.histplot, 'Family_Size', kde=False)

# 類別變數轉換 	•	類別變數轉換 為數值格式（機器學習模型才能處理）。
for col in ['Sex', 'Embarked', 'Pclass', 'Title1', 'Title2', 'Cabin', 'Ticket_info']:
    data[col] = data[col].astype('category').cat.codes

# 建立隨機森林來預測年齡
dataAgeNull = data[data["Age"].isnull()].copy()  # 確保是副本
dataAgeNotNull = data[data["Age"].notnull()]
remove_outlier = dataAgeNotNull[
    (np.abs(dataAgeNotNull["Fare"] - dataAgeNotNull["Fare"].mean()) > (4 * dataAgeNotNull["Fare"].std())) |
    (np.abs(dataAgeNotNull["Family_Size"] - dataAgeNotNull["Family_Size"].mean()) > (4 * dataAgeNotNull["Family_Size"].std()))
]

#預測缺失的 Age 用隨機森林模型來預測 Age 缺失值（根據 Pclass、Sex 等特徵）。
rfModel_age = RandomForestRegressor(n_estimators=2000, random_state=42)
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2', 'Cabin', 'Ticket_info']
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

ageNullValues = rfModel_age.predict(dataAgeNull[ageColumns])
dataAgeNull.loc[:, "Age"] = ageNullValues

# ⚠ append() 被棄用，改用 pd.concat()
data = pd.concat([dataAgeNull, dataAgeNotNull], ignore_index=True)

# 拆分 train 和 test分割訓練集和測試集
dataTrain = data[data['Survived'].notna()].sort_values(by=["PassengerId"])
dataTest = data[data['Survived'].isna()].sort_values(by=["PassengerId"])

# 選擇特徵
features = ['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2', 'Ticket_info', 'Cabin']
dataTrain = dataTrain[['Survived'] + features]
dataTest = dataTest[features]

dataTrain.info()

#訓練 RandomForestClassifier訓練 隨機森林分類模型 
#oob_score_ 用來評估模型準確度。
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)

#生成預測結果 預測測試集 存活結果 儲存為 CSV 以便提交。
rf_res =  rf.predict(dataTest)
submit['Survived'] = rf_res
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)

submit