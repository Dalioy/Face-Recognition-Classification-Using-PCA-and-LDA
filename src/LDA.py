import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_LDA(X_train, y_train):
    import numpy as np

    y_train = np.squeeze(y_train)
    unique_classes = np.unique(y_train)

    class_means = []
    class_sizes = []

    for cls in unique_classes:
        class_data = X_train[y_train == cls]
        class_means.append(np.mean(class_data, axis=0))
        class_sizes.append(len(class_data))

    class_means = np.array(class_means)
    class_sizes = np.array(class_sizes)
    overall_mean = np.mean(X_train, axis=0)

    # Within-class scatter matrix
    S_W = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i, cls in enumerate(unique_classes):
        class_data = X_train[y_train == cls]
        centered = class_data - class_means[i]
        S_W += centered.T @ centered

    S_W += 1e-7 * np.identity(X_train.shape[1])  # Regularization

    # Between-class scatter matrix
    S_B = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(len(unique_classes)):
        diff = class_means[i] - overall_mean
        S_B += class_sizes[i] * np.outer(diff, diff)

    # Solve the eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    projection_matrix = eigvecs[:, :len(unique_classes) - 1]
    return np.real(projection_matrix)

def LDA_projected_data(training_data,test_data,projection_matrix):
    projected_X_train = np.dot(training_data, projection_matrix)
    projected_X_test = np.dot(test_data, projection_matrix)
    return projected_X_train, projected_X_test

def Test_LDA(X_train, X_test, y_train, y_test, LDA_projection_matrix,k):
    projected_X_train, projected_X_test = LDA_projected_data(X_train,X_test,LDA_projection_matrix)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(projected_X_train, y_train.ravel())
    y_pred = knn.predict(projected_X_test)
    accuracy = accuracy_score(y_test, y_pred.ravel())
    return accuracy
