# 🤖 Mini Projet ML - Prétraitement des Données

Un projet complet d'exploration et de comparaison des techniques de mise à l'échelle (scaling) en apprentissage automatique.

## 📋 Structure du Projet

```
mini_projet_ml/
├── src/                          # Code principal
│   ├── preprocessing.py          # Module de fonctions de scaling
│   └── experiments.py            # Expériences avec pipelines
├── notebooks/
│   └── projet.ipynb             # Notebook interactif Jupyter
├── rapport/
│   ├── README.md                # Rapport complet du projet
│   └── rapport.pdf              # Rapport formaté
├── figures/                      # Graphiques et visualisations
├── data/                         # Données du projet
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

## 🚀 Quick Start

### 1. Installation des dépendances

```bash
# Installer les packages
pip install -r requirements.txt

# Ou pour Conda
conda create -n ml-project python=3.10
conda activate ml-project
pip install -r requirements.txt
```

### 2. Exécuter les expériences

```bash
# Run the main experiment script
python src/experiments.py

# Run the preprocessing module tests
python src/preprocessing.py
```

### 3. Ouvrir le Notebook

```bash
# Lancer Jupyter
jupyter notebook notebooks/projet.ipynb

# Ou dans VS Code
# Ouvrir le fichier .ipynb avec l'extension Jupyter
```

---

## 📊 Techniques Comparées

| Technique | Formule | Applications |
|-----------|---------|--------------|
| **StandardScaler** | `z = (x - μ) / σ` | Modèles linéaires, PCA, Neural Nets |
| **MinMaxScaler** | `x' = (x - min) / (max - min)` | Intervalle [0,1], Images |
| **RobustScaler** | `x' = (x - médiane) / IQR` | Données avec outliers |
| **MaxAbsScaler** | `x' = x / max(\|x\|)` | Données creuses (sparse) |

---

## 📈 Résultats Clés

Les expériences montrent que :

✅ **MaxAbsScaler** : Meilleure performance (0.844)  
✅ **RobustScaler** : Bon pour données avec outliers (0.843)  
✅ **StandardScaler** : Choix recommandé par défaut (0.842)  
⚠️ **MinMaxScaler** : Sensible aux valeurs extrêmes (0.834)  
📊 **No Scaling** : Baseline (0.841)

---

## 🔬 Méthodologie

- **Dataset** : Classification binaire avec 1000 échantillons
- **Validation** : Cross-validation (5 folds)
- **Modèle** : Logistic Regression
- **Pipeline** : Évite les fuites de données (data leakage)

---

## 📚 Fichiers Principaux

### `src/preprocessing.py`
Module contenant 4 fonctions de scaling réutilisables :
- `standardize(X)` 
- `minmax_scale(X)`
- `robust_scale(X)`
- `maxabs_scale(X)`

Utilisation :
```python
from preprocessing import standardize
X_scaled = standardize(X)
```

### `src/experiments.py`
Script principal comparant tous les scalers avec :
- Génération automatique du dataset
- Cross-validation (5 folds)
- Tableau des résultats
- Ranking des méthodes

### `notebooks/projet.ipynb`
Notebook Jupyter avec :
- Imports et configuration
- Prétraitement des données
- Visualisations (histogrammes avant/après)
- Analyses interactives

---

## 🛠️ Dépendances

```
- Python 3.10+
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas
- jupyter
```

Voir `requirements.txt` pour les versions exactes.

---

## 📝 Rapport Complet

Le rapport détaillé (`rapport/README.md`) contient :

1. **Introduction** - Contexte et objectifs
2. **Fondements théoriques** - Formules et explications
3. **Méthodologie** - Configuration des expériences
4. **Implémentation** - Code et architecture
5. **Résultats** - Tableaux et graphiques
6. **Discussion** - Interprétation des résultats
7. **Conclusion** - Recommandations

---

## 💡 Recommandations d'Utilisation

### Utiliser StandardScaler quand :
- ✓ Modèles linéaires (Regression, Logistic Regression)
- ✓ SVM
- ✓ PCA
- ✓ Neural Networks (généralement)

### Utiliser RobustScaler quand :
- ✓ Présence d'outliers confirmée
- ✓ Données avec distributions asymétriques
- ✓ Robustesse critique

### Utiliser MinMaxScaler quand :
- ✓ Valeurs bornées nécessaires [0, 1]
- ✓ Images (après normalisation)
- ✓ Réseaux de neurones spécifiques

### Utiliser MaxAbsScaler quand :
- ✓ Données creuses (sparse)
- ✓ Conservation de la sparsité importante
- ✓ Données centrées autour de zéro

---

## 🔗 Ressources

- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [Feature Scaling Best Practices](https://en.wikipedia.org/wiki/Feature_scaling)

---

## 📄 License

Ce projet est fourni à titre éducatif.

---

**Date de création** : 27 Mars 2026  
**Python** : 3.14.3  
**Status** : ✅ Complété
