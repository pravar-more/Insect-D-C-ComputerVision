import os
import pandas as pd

df1 = pd.read_csv('labelSet\Eocanthecona_Bug_A.csv')
df2 = pd.read_csv('labelSet\Larva_Spodoptera_D.csv')
df3 = pd.read_csv('labelSet\Red_Hairy_Catterpillar_C.csv')
df4 = pd.read_csv('labelSet\Tobacco_Caterpillar_B.csv')


# def getPath(x):
#     return os.path.join(x["lable"],x["image"])

df1['path'] = df1['label'] + "/" + df1['image']
df2['path'] = df2['label'] + "/" + df2['image']
df3['path'] = df3['label'] + "/" + df3['image']
df4['path'] = df4['label'] + "/" + df4['image']

# print(df1.head())
frames = [df1,df2,df3,df4]
df = pd.concat(frames)

df.to_csv("MetaData.csv")