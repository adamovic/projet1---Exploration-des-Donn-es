# ==============================
# PREPROCESSING MODULE
# ==============================

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


# ==============================
# 1. Standardization
# ==============================
def standardize(X):
    """
    Standardise les données :
    moyenne = 0, variance = 1
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# ==============================
# 2. Min-Max Scaling
# ==============================
def minmax_scale(X):
    """
    Normalise les données entre 0 et 1
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# ==============================
# 3. Robust Scaling (outliers)
# ==============================
def robust_scale(X):
    """
    Scaling robuste basé sur médiane et IQR
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# ==============================
# 4. MaxAbs Scaling (sparse)
# ==============================
def maxabs_scale(X):
    """
    Scaling adapté aux matrices creuses
    """
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# ==============================
# TEST LOCAL (optionnel)
# ==============================
if __name__ == "__main__":
    # Données test
    X = np.array([[1, 2], [3, 4], [5, 6]])

    print("Données originales:\n", X)

    print("\nStandardScaler:")
    print(standardize(X))

    print("\nMinMaxScaler:")
    print(minmax_scale(X))

    print("\nRobustScaler:")
    print(robust_scale(X))

    print("\nMaxAbsScaler:")
    print(maxabs_scale(X))