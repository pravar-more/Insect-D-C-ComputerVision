import os
import tensorflow as tf
from model import create_model
from DataGenerator import CustomDataGen
import cv2
import pandas as pd

folder = 'dataset'
df = pd.read_csv("dataset\MetaData.csv")
img_ids = df['image'].unique()
train = CustomDataGen(df,img_ids,32,folder)
print(str(train))
model = create_model(input_shape=(256,256,3))

# print(os.path.join(folder, df.loc[df['image'] == '00046IMG_00046_BURST20190912100516.jpg']['path'].to_string()))
# x = list(df.loc[df['image'] == '00046IMG_00046_BURST20190912100516.jpg']['path'])
# print(x[0])
