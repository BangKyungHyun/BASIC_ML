import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import  StandardScaler

from sklearn.datasets import fetch_california_housing
califonia = fetch_california_housing()

df = pd.DataFrame(califonia.data,columns=califonia.feature_names)
df["Target"] = califonia.target
print('df.tail() = \n',df.tail())