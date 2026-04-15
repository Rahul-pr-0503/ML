import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data      
y = iris.target
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Species'] = y
plt.figure(figsize=(6, 4))
species = ['Setosa', 'Versicolor', 'Virginica']
colors = ['brown', 'hotpink', 'purple']
for species, color in zip(np.unique(y), colors):
    plt.scatter(
        df_pca.loc[df_pca['Species'] == species, 'PC1'],
        df_pca.loc[df_pca['Species'] == species, 'PC2'],
        c=color,
        label=iris.target_names[species]
    )
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()