# -*- coding: utf-8 -*-
"""dataFileHndl.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x-WHLeaj4YkJs45E8P8yRqdtZk2AGubH
"""

import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import csv
from bs4 import BeautifulSoup
import glob
import numpy as np





# set search path and glob for files
# here we want to look for csv files in the input directory
path = 'input'
folder = glob.glob(path + '/*.csv')

# create empty list to store dataframes
li = []

# loop through list of files and read each one into a dataframe and append to list
for file in folder:
    # read in csv
    temp_df = pd.read_csv(file)
    # append df to list
    li.append(temp_df)
      #print(f'Successfully created dataframe for {file} with shape {temp_df.shape}')

# concatenate our list of dataframes into one!
df = pd.concat(li, axis=0, ignore_index=True)

filename = "img_data.csv"
    
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)

print(df.shape)
df.head()