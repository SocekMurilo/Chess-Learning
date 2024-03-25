import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from xgboost import XGBRegressor

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
le = LabelEncoder()

moves = df["moves"].str.split()
moves = moves.T
lenthMoves = len(moves)
for sublist in moves:
    for item in sublist:
        df[item] = le.fit_transform(df[item])
print(moves)
print(df.columns)
print (df)
# for col in moves:
#     features = 
# print(features)


        # if item not in df.columns:
        #     df[item] = item
        # else:
        #     if item in df[item].values:
        #         df.loc[len(df), item] = item
        # print(item)



#######################################################################################

# le = LabelEncoder()
# df['victory_status'] = le.fit_transform(df['victory_status'])
# df['winner'] = le.fit_transform(df['winner'])
# df['increment_code'] = le.fit_transform(df['increment_code'])
# df['moves'] = le.fit_transform(df['moves'])
# df['opening_eco'] = le.fit_transform(df['opening_eco'])
# df['opening_name'] = le.fit_transform(df['opening_name'])

# Y = df['white_rating']
# X = df.drop(['black_rating', 'white_rating'], axis = 1)
# X = normalize(X)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# model = RandomForestClassifier()

# model.fit(X_train, Y_train)

# dump(model, 'chess.pkl')

# print("-----------------------------------------------")
# print(mean_absolute_error(Y_train, model.predict(X_train)))
# print(mean_absolute_error(Y_test, model.predict(X_test)))
# print("-----------------------------------------------")


# Ypred = model.predict(X)
# plt.plot(Y)
# plt.plot(Ypred)
# plt.show()