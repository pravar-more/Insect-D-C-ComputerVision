import os
import pandas as pd
import cv2


def img_shape(path):
    img = cv2.imread(os.path.join("dataset", path))
    return img.shape


df1 = pd.read_csv('dataset\Eocanthecona_Bug_A.csv')
df2 = pd.read_csv('dataset\Larva_Spodoptera_D.csv')
df3 = pd.read_csv('dataset\Red_Hairy_Catterpillar_C.csv')
df4 = pd.read_csv('dataset\Tobacco_Caterpillar_B.csv')

df1['path'] = df1['label'] + "\\" + df1['image']
df2['path'] = df2['label'] + "\\" + df2['image']
df3['path'] = df3['label'] + "\\" + df3['image']
df4['path'] = df4['label'] + "\\" + df4['image']


frames = [df1, df2, df3, df4]

df = pd.concat(frames)
df['shape'] = df.apply(lambda x: img_shape(x), axis=1)
df.to_csv("MetaData.csv")
