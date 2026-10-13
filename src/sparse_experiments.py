# ==============================
# SPARSE DATA EXPERIMENTS
# ==============================
# Étude du cas des données creuses (sparse) et 
# l'impact des transformations sur la structure

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import scipy.sparse as sp

print("=" * 60)
print("SPARSE DATA EXPERIMENTS")
print("=" * 60)

# ==============================
# 1. CRÉATION DE DONNÉES TEXTUELLES
# ==============================
texts = [
    "L'intelligence artificielle transforme l'industrie",
    "Le machine learning révolutionne l'analyse de données",
    "Les réseaux de neurones apprennent automatiquement",
    "L'IA et le deep learning changent notre monde",
    "Python est excellent pour le data science",
    "Les algorithmes d'apprentissage sont puissants",
    "Le preprocessing améliore les performances",
    "La feature engineering est cruciale en ML"
]

# Labels pour classification (0: tech, 1: data)
y = np.array([0, 1, 0, 0, 1, 0, 1, 1])

print(f"\nDataset textuel : {len(texts)} documents")
print(f"Labels : {np.unique(y, return_counts=True)}")

# ==============================
# 2. VECTORIZATION TF-IDF
# ==============================
vectorizer = TfidfVectorizer(
    max_features=20,
    stop_words=None,
    lowercase=True,
    ngram_range=(1, 2)
)

X_sparse = vectorizer.fit_transform(texts)

print(f"\nAprès TF-IDF:")
print(f"  Shape : {X_sparse.shape}")
print(f"  Type : {type(X_sparse)}")
print(f"  Densité : {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.4f}")
print(f"  Valeurs non nulles : {X_sparse.nnz}")
print(f"  Min/Max : {X_sparse.data.min():.4f} / {X_sparse.data.max():.4f}")

# ==============================
# 3. COMPARAISON DES TRANSFORMATIONS
# ==============================
print(f"\n{'='*60}")
print("COMPARAISON DES TRANSFORMATIONS SPARSE")
print(f"{'='*60}")

# Configuration des scalers
scalers = {
    "❌ No Scaling": None,
    "✓ MaxAbsScaler": MaxAbsScaler(),
    "✓ StandardScaler (no mean)": StandardScaler(with_mean=False)
}

results = {}

for name, scaler in scalers.items():
    print(f"\nTesting: {name}")
    
    # Pipeline
    if scaler:
        pipe = Pipeline([
            ("scaler", scaler),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ])
    else:
        pipe = Pipeline([
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ])
    
    # Cross-validation (adapté pour petit dataset)
    if len(texts) >= 5:
        scores = cross_val_score(pipe, X_sparse, y, cv=min(3, len(np.unique(y))), scoring='accuracy')
        results[name] = scores
        print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
        print(f"  Mean: {scores.mean():.3f}")
    else:
        print("  Dataset trop petit pour CV")
    
    # Analyse de la structure
    if scaler:
        X_transformed = scaler.fit_transform(X_sparse)
        print(f"  Après transformation:")
        print(f"    Type: {type(X_transformed)}")
        print(f"    Densité: {X_transformed.nnz / (X_transformed.shape[0] * X_transformed.shape[1]):.4f}")
        print(f"    Sparsité conservée: {sp.issparse(X_transformed)}")
        if sp.issparse(X_transformed):
            print(f"    Min/Max: {X_transformed.data.min():.4f} / {X_transformed.data.max():.4f}")

# ==============================
# 4. DÉMONSTRATION DU PROBLÈME DE CENTRAGE
# ==============================
print(f"\n{'='*60}")
print("DÉMONSTRATION : PROBLÈME DU CENTRAGE")
print(f"{'='*60}")

# StandardScaler normal (avec centrage)
scaler_with_mean = StandardScaler(with_mean=True)

try:
    # Tentative de centrage (va densifier)
    X_dense = scaler_with_mean.fit_transform(X_sparse.toarray())
    print(f"\nStandardScaler WITH mean:")
    print(f"  Type après transformation: {type(X_dense)}")
    print(f"  Shape: {X_dense.shape}")
    print(f"  Densité: 1.0000 (PERTE DE SPARSITÉ!)")
    print(f"  Mémoire utilisée: ~{X_dense.nbytes / 1024:.1f} KB")
    
    # Comparaison mémoire
    sparse_memory = X_sparse.data.nbytes + X_sparse.indptr.nbytes + X_sparse.indices.nbytes
    print(f"  Mémoire sparse originale: ~{sparse_memory / 1024:.1f} KB")
    print(f"  Ratio mémoire: {X_dense.nbytes / sparse_memory:.1f}x plus grand!")
    
except Exception as e:
    print(f"Erreur: {e}")

# ==============================
# 5. RECOMMANDATIONS
# ==============================
print(f"\n{'='*60}")
print("RECOMMANDATIONS POUR DONNÉES SPARSE")
print(f"{'='*60}")

recommendations = [
    "✅ Utiliser MaxAbsScaler pour conserver la sparsité",
    "✅ StandardScaler(with_mean=False) si standardisation nécessaire", 
    "❌ JAMAIS StandardScaler(with_mean=True) sur données sparse",
    "✅ Considérer la mémoire et la vitesse de calcul",
    "✅ MaxAbsScaler idéal pour TF-IDF et text mining"
]

for rec in recommendations:
    print(f"  {rec}")

print(f"\n{'='*60}\n")
