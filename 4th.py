import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
iris = load_iris()
X = iris.data      
y = iris.target    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
def cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=False):
    results = {}
    for k in k_values:
        if weighted:
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        else:
            knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[k] = {
            'accuracy': accuracy,
            'f1_score': f1
        }
    return results
k_values = [1, 3, 5, 7, 9]
regular_results = cls_knn(X_train, X_test, y_train, y_test, k_values)
weighted_results = cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=True)
print("Regular k-NN Results:", regular_results)
print("Weighted k-NN Results:", weighted_results)
k_values = [1, 3, 5]
print("Regular k-NN Results:")
regular_results = cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=False)
for k, metrics in regular_results.items():
    print(f"k={k}: Accuracy={metrics['accuracy']:.2f}, F1-Score={metrics['f1_score']:.2f}")
print("\nWeighted k-NN Results:")
weighted_results = cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=True)
for k, metrics in weighted_results.items():
    print(f"k={k}: Accuracy={metrics['accuracy']:.2f}, F1-Score={metrics['f1_score']:.2f}")
print("\nComparison of Regular k-NN and Weighted k-NN:")
for k in k_values:
    regular_acc = regular_results[k]['accuracy']
    weighted_acc = weighted_results[k]['accuracy']
    print(f"k={k}: Regular Accuracy={regular_acc:.2f}, Weighted Accuracy={weighted_acc:.2f}")
best_k_regular = max(regular_results, key=lambda k: regular_results[k]['accuracy'])
best_k_weighted = max(weighted_results, key=lambda k: weighted_results[k]['accuracy'])
print("\nBest k Selection:")
print(f"Best k (Regular k-NN): {best_k_regular} with Accuracy={regular_results[best_k_regular]['accuracy']:.2f}")
print(f"Best k (Weighted k-NN): {best_k_weighted} with Accuracy={weighted_results[best_k_weighted]['accuracy']:.2f}")
final_knn = KNeighborsClassifier(n_neighbors=best_k_regular)
final_knn.fit(X_train, y_train)
final_predictions = final_knn.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
final_f1 = f1_score(y_test, final_predictions, average='weighted')
print("\nFinal Model Performance (Regular k-NN with Best k):")
print(f"Accuracy={final_accuracy:.2f}")
print(f"F1-Score={final_f1:.2f}")