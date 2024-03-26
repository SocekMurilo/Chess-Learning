import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from xgboost import XGBRegressor
from sklearn.cluster import MiniBatchKMeans


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('DataSet\\games.csv')

df.columns = df.columns.str.strip()

df.dropna()

df.drop(
    ['id', 'created_at', 'white_id', 'black_id'],
    axis = 1,
    inplace = True
)


# print(df)

top10 = []
le = LabelEncoder()

print("Comecou!!")

# moves = moves.T
df['moves'] = df["moves"].str.split()
moves = df['moves']
i = 0

for sublist in moves:
    moves = sublist[0: 10: ]
    for item in sublist:
        if item in df.loc[i, 'moves']:
            df.loc[i, item] = (int)(1)
    i += 1

df = df.fillna((int)(0))

df.drop(
    [ 'moves'],
    axis = 1,
    inplace = True
)
print(df)
print(df["e4"])
#######################################################################################

df['victory_status'] = le.fit_transform(df['victory_status'])
df['winner'] = le.fit_transform(df['winner'])
df['increment_code'] = le.fit_transform(df['increment_code'])
# df['moves'] = le.fit_transform(df['moves'])
df['opening_eco'] = le.fit_transform(df['opening_eco'])
df['opening_name'] = le.fit_transform(df['opening_name'])

Y = df['winner']
X = df.drop(['winner'], axis = 1)
X = normalize(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

model = MiniBatchKMeans(n_clusters=10,
                        random_state=0,
                        batch_size=6,
                        n_init="auto", )

model.fit(X_train, Y_train)

dump(model, 'chess.csv')

print("-----------------------------------------------")
print(mean_absolute_error(Y_train, model.predict(X_train)))
print(mean_absolute_error(Y_test, model.predict(X_test)))
print("-----------------------------------------------")


Ypred = model.fit_predict(X)
plt.plot(Y)
plt.plot(Ypred)
plt.show()