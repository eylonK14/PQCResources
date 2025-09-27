import pandas as pd
import ntpath

df = pd.read_csv('all-files-nech-lab.csv')
df['3'] = '[0, 0, 0]'
df['4'] = '[0, 0, 0]'
df = df.drop(columns=['Unnamed: 0'])
df.to_csv('all-files-nech-lab-zero.csv')