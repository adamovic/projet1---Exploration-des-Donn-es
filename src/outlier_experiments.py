# ==============================
# OUTLIER EXPERIMENTS
# ==============================
# Étude de l'impact des valeurs aberrantes sur les 
# différentes techniques de scaling

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import seaborn as sns

print("=" * 60)
print("OUTLIER IMPACT EXPERIMENTS")
print("=" * 60)

# ==============================
# 1. CRÉATION DE DONNÉES AVEC OUTLIERS
# ==============================
# Dataset de base
X_base, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_clusters_per_class=1
)

# Ajout de outliers extrêmes
X_with_outliers = X_base.copy()
outlier_indices = np.random.choice(X_base.shape[0], 20, replace=False)

# Ajouter des outliers extrêmes sur la première feature
X_with_outliers[outlier_indices, 0] += np.random.choice([-50, 50], 20)

# Ajouter des outliers modérés sur la deuxième feature  
X_with_outliers[outlier_indices, 1] += np.random.choice([-10, 10], 20)

print(f"\nDataset créé:")
print(f"  Échantillons: {X_with_outliers.shape[0]}")
print(f"  Features: {X_with_outliers.shape[1]}")
print(f"  Outliers ajoutés: {len(outlier_indices)}")
print(f"  Range original: [{X_base.min():.2f}, {X_base.max():.2f}]")
print(f"  Range avec outliers: [{X_with_outliers.min():.2f}, {X_with_outliers.max():.2f}]")

# ==============================
# 2. VISUALISATION DES DISTRIBUTIONS
# ==============================
print(f"\n{'='*60}")
print("VISUALISATION DES DISTRIBUTIONS")
print(f"{'='*60}")

scalers = {
    "Original": None,
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler()
}

# Créer les visualisations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, scaler) in enumerate(scalers.items()):
    ax = axes[idx]
    
    if scaler:
        X_transformed = scaler.fit_transform(X_with_outliers)
    else:
        X_transformed = X_with_outliers
    
    # Histogramme pour la première feature (avec outliers)
    ax.hist(X_transformed[:, 0], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title(f'{name}\nFeature 0 (avec outliers)')
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Fréquence')
    ax.grid(True, alpha=0.3)
    
    # Statistiques
    mean_val = np.mean(X_transformed[:, 0])
    std_val = np.std(X_transformed[:, 0])
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val + std_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val - std_val:.2f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/outlier_distributions.png', dpi=150, bbox_inches='tight')
print("✅ Graphique sauvegardé: figures/outlier_distributions.png")

# ==============================
# 3. BOXPLOTS COMPARATIFS
# ==============================
plt.figure(figsize=(12, 8))

data_for_boxplot = []
labels_for_boxplot = []

for name, scaler in scalers.items():
    if scaler:
        X_transformed = scaler.fit_transform(X_with_outliers)
    else:
        X_transformed = X_with_outliers
    
    # Ajouter les deux features
    data_for_boxplot.append(X_transformed[:, 0])
    data_for_boxplot.append(X_transformed[:, 1])
    labels_for_boxplot.extend([f'{name}_F0', f'{name}_F1'])

plt.boxplot(data_for_boxplot, labels=labels_for_boxplot)
plt.title('Comparaison des Boxplots - Impact des Outliers par Scaling Technique')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Valeur')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/outlier_boxplots.png', dpi=150, bbox_inches='tight')
print("✅ Boxplot sauvegardé: figures/outlier_boxplots.png")

# ==============================
# 4. IMPACT SUR LES PERFORMANCES DES MODÈLES
# ==============================
print(f"\n{'='*60}")
print("IMPACT SUR LES PERFORMANCES DES MODÈLES")
print(f"{'='*60}")

# Comparaison sur données sans outliers vs avec outliers
datasets = {
    "Sans Outliers": X_base,
    "Avec Outliers": X_with_outliers
}

all_results = {}

for dataset_name, X_dataset in datasets.items():
    print(f"\n--- Dataset: {dataset_name} ---")
    
    results = {}
    
    for scaler_name, scaler in scalers.items():
        if scaler_name == "Original":
            continue  # Skip original for model comparison
            
        if scaler:
            pipe = Pipeline([
                ("scaler", scaler),
                ("model", LogisticRegression(max_iter=1000, random_state=42))
            ])
        else:
            pipe = Pipeline([
                ("model", LogisticRegression(max_iter=1000, random_state=42))
            ])
        
        scores = cross_val_score(pipe, X_dataset, y, cv=5, scoring='accuracy')
        results[scaler_name] = scores
        
        print(f"  {scaler_name:<15}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    all_results[dataset_name] = results

# ==============================
# 5. ANALYSE DE LA ROBUSTESSE
# ==============================
print(f"\n{'='*60}")
print("ANALYSE DE LA ROBUSTESSE AUX OUTLIERS")
print(f"{'='*60}")

# Calculer la dégradation de performance
print("Dégradation due aux outliers (pourcentage):")
print("-" * 50)

for scaler_name in ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler"]:
    score_without = all_results["Sans Outliers"][scaler_name].mean()
    score_with = all_results["Avec Outliers"][scaler_name].mean()
    degradation = (score_without - score_with) / score_without * 100
    
    print(f"{scaler_name:<15}: -{degradation:.2f}% ({score_without:.3f} → {score_with:.3f})")

# ==============================
# 6. STATISTIQUES DÉTAILLÉES
# ==============================
print(f"\n{'='*60}")
print("STATISTIQUES DÉTAILLÉES PAR TRANSFORMATION")
print(f"{'='*60}")

def print_detailed_stats(X, name):
    print(f"\n{name}:")
    print(f"  Moyenne: {np.mean(X, axis=0)}")
    print(f"  Écart-type: {np.std(X, axis=0)}")
    print(f"  Médiane: {np.median(X, axis=0)}")
    print(f"  Min: {np.min(X, axis=0)}")
    print(f"  Max: {np.max(X, axis=0)}")
    print(f"  Q1: {np.percentile(X, 25, axis=0)}")
    print(f"  Q3: {np.percentile(X, 75, axis=0)}")
    print(f"  IQR: {np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)}")

for scaler_name, scaler in scalers.items():
    if scaler:
        X_transformed = scaler.fit_transform(X_with_outliers)
        print_detailed_stats(X_transformed, scaler_name)
    else:
        print_detailed_stats(X_with_outliers, "Original")

# ==============================
# 7. RECOMMANDATIONS
# ==============================
print(f"\n{'='*60}")
print("RECOMMANDATIONS FINALES")
print(f"{'='*60}")

recommendations = [
    "🎯 StandardScaler: Bon choix général, mais sensible aux outliers",
    "⚠️  MinMaxScaler: À éviter avec présence d'outliers extrêmes",
    "🛡️  RobustScaler: Meilleur choix confirmé avec outliers",
    "⚡ MaxAbsScaler: Alternative robuste, conserve les signes",
    "",
    "📊 En pratique:",
    "  - Analyser toujours la distribution avant scaling",
    "  - Utiliser des visualisations (boxplots, histograms)",
    "  - Tester plusieurs approches sur validation croisée",
    "  - Privilégier RobustScaler si outliers détectés"
]

for rec in recommendations:
    print(f"  {rec}")

print(f"\n{'='*60}\n")

# Sauvegarder une figure récapitulative
plt.show()
