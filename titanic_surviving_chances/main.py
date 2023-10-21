import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

data = pd.read_excel('titanic.xls')
data.drop(['name','boat','body','cabin', 'fare','sibsp','parch','ticket','embarked','home.dest'], axis=1,inplace=True)
data.dropna(axis=0, inplace=True)
described_data = data.describe()
stats = data.groupby(['sex','pclass']).mean()

age = int(input('Quel est votre age ?'))
sexe = input("quel est votre sexe ?")







