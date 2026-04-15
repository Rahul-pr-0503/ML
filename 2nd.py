import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = sns.load_dataset('iris')
x_col = 'sepal_length'
y_col = 'petal_length'
correlation = data[['sepal_length', 'petal_length']].corr('pearson')
print("Pearson Correlation Coefficient:\n", correlation)
covariance = data[['sepal_length', 'petal_length']].cov()
print("Covariance Matrix:\n", covariance)
plt.figure(figsize=(8, 5))
plt.scatter(data[x_col], data[y_col])
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f"Scatter Plot of {x_col} vs {y_col}")
plt.show()
data_co = data.iloc[:, :-1]
covariance_matrix = data_co.cov()
correlation_matrix = data_co.corr()
print("Covariance Matrix:\n", covariance_matrix)
print("\nCorrelation Matrix:\n", correlation_matrix)
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()