import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("titanic.csv")
num_col = "Age" 
data = df[num_col].dropna()
print("Mean:", data.mean())
print("Median:", data.median())
print("Mode:", data.mode()[0])
print("Standard Deviation:", data.std())
print("Variance:", data.var())
print("Range:", data.max() - data.min())
plt.hist(data, bins=10)
plt.title("Histogram of " + num_col)
plt.show()
plt.boxplot(data)
plt.title("Boxplot of " + num_col)
plt.show()
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = data[(data < lower) | (data > upper)]
print("Outliers:\n", outliers)
cat_col = "Sex" 
freq = df[cat_col].value_counts()
print("\nCategory Frequency:\n", freq)
freq.plot(kind='bar')
plt.title("Bar Chart of " + cat_col)
plt.show()
