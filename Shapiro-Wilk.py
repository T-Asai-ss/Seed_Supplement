# Investigation on the correspondence between randomness of output due to seed value in in silico models and sampling frequency in biological tests
# Python code of Shapiro-Wilk test

# ## import library
# In[1]:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ## input data
# In[2]:
df = pd.DataFrame(pd.read_csv(filepath_or_buffer="./tp2_100.csv", encoding="ms932", sep=","))
row1 = df.iloc[0].tolist()
row2 = df.iloc[1].tolist()
row3 = df.iloc[2].tolist()
row4 = df.iloc[3].tolist()
row5 = df.iloc[4].tolist()
row6 = df.iloc[5].tolist()
row7 = df.iloc[6].tolist()
row8 = df.iloc[7].tolist()
row9 = df.iloc[8].tolist()
row10 = df.iloc[9].tolist()
row11 = df.iloc[10].tolist()
row12 = df.iloc[11].tolist()
row13 = df.iloc[12].tolist()
row14 = df.iloc[13].tolist()
row15 = df.iloc[14].tolist()
row16 = df.iloc[15].tolist()
row17 = df.iloc[16].tolist()
row18 = df.iloc[17].tolist()
row19 = df.iloc[18].tolist()
row20 = df.iloc[19].tolist()
row21 = df.iloc[20].tolist()
row22 = df.iloc[21].tolist()

# ## shapiro
# In[3]:
from scipy.stats import shapiro
shapiro(row1)
shapiro(row2)
shapiro(row3)
shapiro(row4)
shapiro(row5)
shapiro(row6)
shapiro(row7)
shapiro(row8)
shapiro(row9)
shapiro(row10)
shapiro(row11)
shapiro(row12)
shapiro(row13)
shapiro(row14)
shapiro(row15)
shapiro(row16)
shapiro(row17)
shapiro(row18)
shapiro(row19)
shapiro(row20)
shapiro(row21)
shapiro(row22)
