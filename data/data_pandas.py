
import numpy as np
import pandas as pd
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)

poke = pd.read_csv(train_file_path)

#poke = pd.read_csv('train_path.txt', delimiter='\t')
#poke = pd.read_excel('train_path.xlsx')

print(poke.head())
print(poke.tail(3))

# print columns
print(poke.columns)
print(poke['alone'][:5])
print(poke[['alone', 'sex', 'age']][:5])

# print rows
print(poke.iloc[1:4])
print(poke.iloc[1,5])

for index, row in poke.iterrows():
    print(index, row['age'])

# filters
male = poke.loc[poke['sex'] == 'male']
print(male)

# describe
print(poke.describe())

# sorting
poke.sort_values('age', ascending=False)
poke.sort_values(['age', 'fare'], ascending=[1, 0])

# making changes to the data
poke['total'] = poke['age'] + poke['parch']
print(poke.head())

poke.drop(columns=['total'])

poke['total'] = poke.iloc[:, 2:4].sum(axis=1)
print(poke.head())

# save modified data to files
poke.to_csv('modified.csv', index=True)
poke.to_excel('modified.xlsx', index=False)
poke.to_csv('modified.txt', index=False, sep='\t')

# filters
filtered = poke.loc[(poke['sex']=='male') & (poke['age'] > 30)]
filtered = filtered.reset_index()

print(filtered)

# contains function is case-sensitive
place_filter = poke.loc[~poke['embark_town'].str.contains('South')]  
print(place_filter)

import re
place_filter = poke.loc[~poke['embark_town'].str.contains('^So[a-z]*', flags=re.I, regex=True)] 
print(place_filter)

# aggregate statistics (groupby)
print(poke.groupby(['embark_town']).mean())
print(poke.groupby(['embark_town', 'sex']).count())


# working with large amount of data
# read 5 rows a time
new_df = pd.DataFrame(columns=poke.columns)
for df in pd.read_csv('modified.csv', chunksize=5):
    # print(df)
    new_df = pd.concat([new_df, df])

print(new_df.describe())