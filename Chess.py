import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns
from warnings import simplefilter


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

df = pd.read_csv('DataSet\\games.csv')

df.columns = df.columns.str.strip()
df.dropna()
df.drop(
    ['id', 'created_at', 'white_id', 'black_id', 'increment_code', 'opening_eco', "last_move_at", "rated", "opening_name", "black_rating", "opening_ply"],
    axis = 1,
    inplace = True
)

le = LabelEncoder()

print("Comecou!!")

df['moves'] = df["moves"].str.split()
moves = df['moves']
i = 0

for sublist in moves:
    moves = sublist[0: 10: 2]
    for item in sublist:
        if item in df.loc[i, 'moves']:
            df.loc[i, item] = 1
    i += 2
    if i > 20056:
        print("Terminou!!")
        break
df = df.fillna((int)(0))

df.drop(
    [ 'moves'],
    axis = 1,
    inplace = True
)

df['victory_status'] = le.fit_transform(df['victory_status'])
df['winner'] = le.fit_transform(df['winner'])
print(le.classes_)

Y = df['winner']
X = df

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = MiniBatchKMeans(n_clusters=6,
                        random_state=42,
                        batch_size=20,
                        max_iter=500,
                        n_init="auto")
model.fit(X_train, X_test)
dump(model, 'chess.pkl')

preds = model.predict(X)
df['cluster'] = preds

data = list(zip(preds, Y, df['d4'].values.tolist()))

win = 0
draw = 0
lost = 0

for i in data:
    if i[2] == 1:
        if i[0] == 0:
            if i[1] == 2:
                win += 1
            elif i[1] == 1:
                draw += 1
            else:
                lost += 1

total = win + draw + lost
porcent = draw / total
        
print(porcent)
print("Acabou!!")

print(list(zip(df.columns, model.cluster_centers_[0])))
print(list(zip(df.columns, model.cluster_centers_[1])))
print(list(zip(df.columns, model.cluster_centers_[2])))
print(list(zip(df.columns, model.cluster_centers_[3])))
print(list(zip(df.columns, model.cluster_centers_[4])))
print(list(zip(df.columns, model.cluster_centers_[5])))

plt.figure(figsize=(10, 8))
sns.set_theme(style="ticks")
sns.pairplot(df.sample(1000), vars=df.columns[0:8], hue='cluster', palette='viridis')
plt.suptitle('Matriz de Dispersão (Scatter Matrix) para as primeiras 5 variáveis', y=1.02)
plt.show()
