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
    ['id', 'created_at', 'white_id', 'black_id', 'increment_code', 'opening_eco'],
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
        break
df = df.fillna((int)(0))

df.drop(
    [ 'moves'],
    axis = 1,
    inplace = True
)

df['victory_status'] = le.fit_transform(df['victory_status'])
df['winner'] = le.fit_transform(df['winner'])
df['opening_name'] = le.fit_transform(df['opening_name'])

Y = df['winner']
X = df.drop(['winner'], axis = 1)

model = MiniBatchKMeans(n_clusters=6,
                        random_state=5464,
                        batch_size=40,
                        n_init="auto")

model.fit(X)
dump(model, 'chess.pkl')

preds = model.predict(X)
df['cluster'] = preds

X_train, X_test, y_train, y_test = train_test_split(X, preds, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

plt.figure(figsize=(10, 8))
sns.set_theme(style="ticks")
sns.pairplot(df.sample(1000), vars=df.columns[4:12], hue='cluster', palette='viridis')
plt.suptitle('Matriz de Dispersão (Scatter Matrix) para as primeiras 5 variáveis', y=1.02)
plt.show()