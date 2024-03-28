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
    moves = sublist[0: 10: 2]
    for item in sublist:
        if item in df.loc[i, 'moves']:
            df.loc[i, item] = 1
    i += 2
    if i > 20056:
        break


rows = df.shape[0] 
cols = df.shape[1] 
print("Rows: " + str(rows)) 
print("Columns: " + str(cols)) 

df = df.fillna((int)(0))


df.drop(
    [ 'moves'],
    axis = 1,
    inplace = True
)
#######################################################################################

df['victory_status'] = le.fit_transform(df['victory_status'])
df['winner'] = le.fit_transform(df['winner'])
df['increment_code'] = le.fit_transform(df['increment_code'])
df['opening_eco'] = le.fit_transform(df['opening_eco'])
df['opening_name'] = le.fit_transform(df['opening_name'])

Y = df['winner']
X = df.drop(['white_rating', 'white_rating'], axis = 1)

model = MiniBatchKMeans(n_clusters=6,
                        random_state=5464,
                        batch_size=20,
                        n_init="auto" )
model.fit(X)
dump(model, 'chess.pkl')

preds = model.predict(X)
for index in df.index:
    df.loc[index, "cluster"] = preds[index]

print(df['cluster'])
centroids = model.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]

df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2], 3:cen_x[3], 4:cen_x[4], 5:cen_x[5]})
df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2], 3:cen_y[3], 4:cen_y[4], 5:cen_y[5]})

colors = ['#DF2020', '#81DF20', '#2095DF', '#677DB7', '#191308', '#21FA90']
df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3], 4:colors[4], 5:colors[5]})

plt.scatter(df['cluster'], df['winner'], alpha = 0.6, s=10, cmap='viridis')
