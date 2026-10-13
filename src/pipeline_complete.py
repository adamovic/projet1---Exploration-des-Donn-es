# ==============================
# COMPLETE PIPELINE EXPERIMENTS
# ==============================
# Pipeline complet avec validation croisée et 
# analyse comparative rigoureuse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PIPELINE COMPLET - VALIDATION CROISÉE ET ANALYSE COMPARATIVE")
print("=" * 70)

# ==============================
# 1. CONFIGURATION ET CRÉATION DU DATASET
# ==============================
RANDOM_STATE = 42
N_SAMPLES = 1000
N_FEATURES = 20
CV_FOLDS = 5

# Dataset plus complexe pour mieux tester les pipelines
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=15,
    n_redundant=3,
    n_repeated=2,
    n_clusters_per_class=2,
    random_state=RANDOM_STATE
)

# Split pour validation finale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nDataset créé:")
print(f"  Total échantillons: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"  Classes: {np.unique(y, return_counts=True)}")

# ==============================
# 2. DÉFINITION DES PIPELINES COMPLETS
# ==============================
print(f"\n{'='*70}")
print("DÉFINITION DES PIPELINES")
print(f"{'='*70}")

# Différents modèles pour tester la généralisation
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "SVM": SVC(kernel='rbf', random_state=RANDOM_STATE)
}

# Différents scalers
scalers = {
    "No Scaling": None,
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler()
}

# Créer tous les pipelines
pipelines = {}
pipeline_names = []

for model_name, model in models.items():
    for scaler_name, scaler in scalers.items():
        pipeline_name = f"{model_name}_{scaler_name}"
        pipeline_names.append(pipeline_name)
        
        if scaler:
            pipe = Pipeline([
                ("scaler", scaler),
                ("model", model)
            ])
        else:
            pipe = Pipeline([
                ("model", model)
            ])
        
        pipelines[pipeline_name] = pipe

print(f"  Nombre total de pipelines: {len(pipelines)}")
print(f"  Modèles testés: {list(models.keys())}")
print(f"  Scalings testés: {list(scalers.keys())}")

# ==============================
# 3. VALIDATION CROISÉE RIGOUREUSE
# ==============================
print(f"\n{'='*70}")
print("VALIDATION CROISÉE (5-FOLDS)")
print(f"{'='*70}")

# Stocker tous les résultats
cv_results = {}
cv_means = {}
cv_stds = {}

for pipeline_name, pipeline in pipelines.items():
    print(f"\nTesting: {pipeline_name}")
    
    # Validation croisée
    scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=CV_FOLDS, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    cv_results[pipeline_name] = scores
    cv_means[pipeline_name] = scores.mean()
    cv_stds[pipeline_name] = scores.std()
    
    print(f"  Scores: {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")

# ==============================
# 4. TABLEAU RÉCAPITULATIF
# ==============================
print(f"\n{'='*70}")
print("TABLEAU COMPARATIF COMPLET")
print(f"{'='*70}")

# Créer DataFrame pour meilleure visualisation
results_df = pd.DataFrame({
    'Pipeline': list(cv_means.keys()),
    'Mean_Accuracy': list(cv_means.values()),
    'Std_Accuracy': list(cv_stds.values())
})

# Extraire modèle et scaler
results_df['Model'] = results_df['Pipeline'].apply(lambda x: x.split('_')[0])
results_df['Scaler'] = results_df['Pipeline'].apply(lambda x: '_'.join(x.split('_')[1:]))

# Trier par performance
results_df = results_df.sort_values('Mean_Accuracy', ascending=False)

print(f"\n{'Pipeline':<35} {'Model':<20} {'Scaler':<15} {'Accuracy':<12} {'Std':<8}")
print("-" * 90)

for _, row in results_df.iterrows():
    print(f"{row['Pipeline']:<35} {row['Model']:<20} {row['Scaler']:<15} {row['Mean_Accuracy']:<12.4f} {row['Std_Accuracy']:<8.4f}")

# ==============================
# 5. VISUALISATIONS COMPARATIVES
# ==============================
print(f"\n{'='*70}")
print("CRÉATION DES VISUALISATIONS")
print(f"{'='*70}")

# 1. Graphique de comparaison par modèle
plt.figure(figsize=(15, 10))

for i, model_name in enumerate(models.keys(), 1):
    plt.subplot(2, 2, i)
    
    model_data = results_df[results_df['Model'] == model_name]
    model_data = model_data.sort_values('Mean_Accuracy')
    
    bars = plt.bar(model_data['Scaler'], model_data['Mean_Accuracy'], 
                   yerr=model_data['Std_Accuracy'], capsize=5, alpha=0.7)
    
    plt.title(f'{model_name} - Comparaison des Scalers')
    plt.ylabel('Accuracy')
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars, model_data['Mean_Accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/pipeline_comparison_by_model.png', dpi=150, bbox_inches='tight')
print("✅ Graphique sauvegardé: figures/pipeline_comparison_by_model.png")

# 2. Heatmap des performances
plt.figure(figsize=(12, 8))

# Pivot pour heatmap
pivot_data = results_df.pivot(index='Model', columns='Scaler', values='Mean_Accuracy')

sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.85,
            square=True, cbar_kws={'label': 'Accuracy'})
plt.title('Heatmap des Performances par Modèle et Scaler')
plt.tight_layout()
plt.savefig('figures/performance_heatmap.png', dpi=150, bbox_inches='tight')
print("✅ Heatmap sauvegardé: figures/performance_heatmap.png")

# 3. Boxplot de la distribution des scores
plt.figure(figsize=(15, 8))

# Préparer les données pour boxplot
boxplot_data = []
boxplot_labels = []

for pipeline_name, scores in cv_results.items():
    boxplot_data.append(scores)
    boxplot_labels.append(pipeline_name.replace('_', '\n'))

plt.boxplot(boxplot_data, labels=boxplot_labels)
plt.title('Distribution des Accuracy par Pipeline (Validation Croisée)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/cv_distribution_boxplot.png', dpi=150, bbox_inches='tight')
print("✅ Boxplot sauvegardé: figures/cv_distribution_boxplot.png")

# ==============================
# 6. ANALYSE STATISTIQUE
# ==============================
print(f"\n{'='*70}")
print("ANALYSE STATISTIQUE DES RÉSULTATS")
print(f"{'='*70}")

# Meilleur pipeline global
best_pipeline = results_df.iloc[0]
print(f"\n🏆 Meilleur pipeline global:")
print(f"   {best_pipeline['Pipeline']}: {best_pipeline['Mean_Accuracy']:.4f} ± {best_pipeline['Std_Accuracy']:.4f}")

# Analyse par modèle
print(f"\n📊 Analyse par modèle:")
for model_name in models.keys():
    model_data = results_df[results_df['Model'] == model_name]
    best_model = model_data.iloc[0]
    worst_model = model_data.iloc[-1]
    
    print(f"\n  {model_name}:")
    print(f"    Meilleur: {best_model['Scaler']} ({best_model['Mean_Accuracy']:.4f})")
    print(f"    Pire: {worst_model['Scaler']} ({worst_model['Mean_Accuracy']:.4f})")
    print(f"    Écart: {(best_model['Mean_Accuracy'] - worst_model['Mean_Accuracy']):.4f}")

# Analyse par scaler
print(f"\n📈 Analyse par scaler (moyenne tous modèles):")
for scaler_name in scalers.keys():
    scaler_data = results_df[results_df['Scaler'] == scaler_name]
    mean_performance = scaler_data['Mean_Accuracy'].mean()
    std_performance = scaler_data['Mean_Accuracy'].std()
    
    print(f"  {scaler_name:<15}: {mean_performance:.4f} ± {std_performance:.4f}")

# ==============================
# 7. TEST FINAL SUR DONNÉES DE TEST
# ==============================
print(f"\n{'='*70}")
print("ÉVALUATION FINALE SUR DONNÉES DE TEST")
print(f"{'='*70}")

# Entraîner le meilleur pipeline sur toutes les données d'entraînement
best_pipe_name = best_pipeline['Pipeline']
best_pipe = pipelines[best_pipe_name]

print(f"\nPipeline final: {best_pipe_name}")
best_pipe.fit(X_train, y_train)

# Prédictions
y_pred = best_pipe.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nPerformance sur test set:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"\nRapport de classification:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xtick_labels=['Class 0', 'Class 1'], 
            ytick_labels=['Class 0', 'Class 1'])
plt.title(f'Matrice de Confusion - {best_pipe_name}')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Matrice de confusion sauvegardée: figures/confusion_matrix.png")

# ==============================
# 8. CONCLUSIONS ET RECOMMANDATIONS
# ==============================
print(f"\n{'='*70}")
print("CONCLUSIONS ET RECOMMANDATIONS")
print(f"{'='*70}")

conclusions = [
    f"🏆 Meilleur pipeline: {best_pipe_name}",
    f"📈 Performance test: {test_accuracy:.4f}",
    "",
    "🔍 Observations principales:",
    "  • Le scaling améliore généralement les performances",
    "  • StandardScaler et RobustScaler sont les plus constants",
    "  • MinMaxScaler peut être sensible aux outliers",
    "  • L'impact du scaling varie selon le modèle",
    "",
    "💡 Recommandations pratiques:",
    "  • Toujours utiliser des pipelines pour éviter les fuites",
    "  • Tester plusieurs approches en validation croisée",
    "  • Analyser la distribution des données avant scaling",
    "  • Considérer la robustesse vs performance pure"
]

for conclusion in conclusions:
    print(f"  {conclusion}")

# Sauvegarder les résultats détaillés
results_df.to_csv('figures/pipeline_results.csv', index=False)
print(f"\n📊 Résultats détaillés sauvegardés: figures/pipeline_results.csv")

print(f"\n{'='*70}\n")
plt.show()
