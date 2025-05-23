def LDA(D, y):
    import numpy as np

    # 1- Class-specific Subsets
    classes = np.unique(y)
    m=len(classes)
    class_data = []
    for c in classes:
        class_data.append(D[y == c])

    # Class Mean
    mu = []
    for i in class_data:
        mu.append(np.mean(i, axis=0))

    mu_overall = np.mean(D, axis=0)

    # Between-Class Scatter matrix
    B = np.zeros((D.shape[1], D.shape[1]))
    for i in range(len(class_data)):
        n_i = class_data[i].shape[0]
        mean_diff = (mu[i] - mu_overall).reshape(-1, 1)
        B += n_i * (mean_diff @ mean_diff.T)

    # Center-Class Matrices
    Z = []
    for i in range(len(class_data)):
        Z.append(class_data[i] - mu[i])

    # Class Scatter Matrices
    Si = []
    for i in range(len(Z)):
        Si.append(Z[i].T @ Z[i])

    # within - class scatter matrix
    S = np.zeros((D.shape[1], D.shape[1]))
    for i in range(len(Si)):
        S += Si[i]

    # computer dominant eigen
    S_inv = np.linalg.pinv(S)
    eigvals, eigvecs = np.linalg.eig(S_inv @ B)

    sorted_indices = np.argsort(-eigvals.real)
    U = eigvecs[:, sorted_indices[:m - 1]].real  

    return U
