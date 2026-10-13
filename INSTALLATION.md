# 📦 Guide d'Installation et d'Utilisation

## 🚀 Installation Rapide

### Prérequis
- Python 3.10+ (recommandé : 3.14.3)
- Git (pour cloner si nécessaire)

### 1. Cloner le Projet
```bash
git clone <repository-url>
cd mini_projet_ml
```

### 2. Créer l'Environnement Virtuel
```bash
# Avec venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Avec conda
conda create -n ml-project python=3.14
conda activate ml-project
```

### 3. Installer les Dépendances
```bash
pip install -r requirements.txt
```

### 4. Vérifier l'Installation
```bash
python -c "import sklearn; print('scikit-learn version:', sklearn.__version__)"
```

---

## 📁 Structure du Projet

```
mini_projet_ml/
├── src/                          # Code source
│   ├── main.py                   # Programme principal
│   ├── experiments.py            # Expériences de base
│   ├── sparse_experiments.py      # Cas données creuses
│   ├── outlier_experiments.py    # Analyse outliers
│   ├── pipeline_complete.py      # Pipeline complet
│   ├── preprocessing.py         # Module de fonctions
│   └── generate_visualizations.py # Générateur de graphiques
├── notebooks/
│   └── projet.ipynb             # Notebook Jupyter
├── figures/                      # Graphiques générés
│   ├── scaling_comparison.png
│   ├── performance_heatmap.png
│   ├── outlier_boxplots.png
│   ├── sparse_analysis.png
│   ├── convergence_analysis.png
│   └── summary_dashboard.png
├── rapport/
│   ├── README.md                 # Rapport détaillé
│   └── RAPPORT_FINAL.md          # Rapport final formatté
├── data/                         # Données (générées)
├── requirements.txt              # Dépendances Python
├── INSTALLATION.md               # Ce guide
├── PRESENTATION.md               # Support présentation
└── README.md                     # Description projet
```

---

## 🎯 Utilisation Rapide

### Lancer le Programme Principal
```bash
cd src/
python main.py
```

**Menu disponible :**
1. 📊 Expériences principales de scaling
2. 🌿 Cas des données creuses (sparse)
3. 🎯 Analyse des outliers
4. 🔄 Pipeline complet avec validation croisée
5. 🔧 Tests des fonctions de prétraitement
6. 🚀 Exécuter toutes les expériences
7. 📈 Générer les visualisations
0. 🚪 Quitter

### Exécuter une Expérience Spécifique
```bash
# Expériences de base
python experiments.py

# Cas des données sparse
python sparse_experiments.py

# Analyse des outliers
python outlier_experiments.py

# Pipeline complet
python pipeline_complete.py

# Tests des fonctions
python preprocessing.py

# Générer toutes les visualisations
python generate_visualizations.py
```

---

## 📊 Résultats Attendus

### Performances Typiques
```
RANKING (by mean accuracy)
============================================================
1. MaxAbsScaler                 0.8440  ↑ +0.30%
2. RobustScaler                 0.8430  ↑ +0.20%
3. StandardScaler               0.8420  ↑ +0.10%
4. No Scaling (baseline)        0.8410  → +0.00%
5. MinMaxScaler                 0.8340  → -0.70%
```

### Visualisations Générées
- `scaling_comparison.png` : Comparaison des distributions
- `performance_heatmap.png` : Heatmap des performances
- `outlier_boxplots.png` : Impact des outliers
- `sparse_analysis.png` : Analyse données sparse
- `convergence_analysis.png` : Convergence des algorithmes
- `summary_dashboard.png` : Tableau de bord récapitulatif

---

## 🔧 Configuration

### Modifier les Paramètres
```python
# Dans experiments.py ou autres fichiers
RANDOM_STATE = 42      # Reproductibilité
N_SAMPLES = 1000       # Taille du dataset
CV_FOLDS = 5           # Nombre de folds CV
MAX_ITER = 1000        # Itérations max LogisticRegression
```

### Ajouter de Nouveaux Modèles
```python
# Dans pipeline_complete.py
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='rbf'),
    "VotreModele": VotreClassifier()  # Ajouter ici
}
```

---

## 🐛 Dépannage

### Problèmes Courants

#### 1. ModuleNotFoundError
```bash
# Solution : Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

#### 2. Problèmes de Visualisation
```bash
# Installer les dépendances manquantes
pip install matplotlib seaborn
```

#### 3. Problèmes de Mémoire
```bash
# Réduire la taille du dataset
N_SAMPLES = 500  # Au lieu de 1000
```

#### 4. Problèmes de Convergence
```bash
# Augmenter le nombre d'itérations
MAX_ITER = 2000  # Au lieu de 1000
```

### Vérification de l'Installation
```python
# Script de test
import numpy as np
import sklearn
import matplotlib
import pandas as pd

print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Pandas: {pd.__version__}")
```

---

## 📚 Pour Aller Plus Loin

### Extensions Possibles
1. **Deep Learning** : Tester avec PyTorch/TensorFlow
2. **AutoML** : Intégrer auto-sklearn
3. **Big Data** : Adapter pour Spark ML
4. **Déploiement** : Créer une API Flask/FastAPI

### Ressources Externes
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Towards Data Science](https://towardsdatascience.com/)

---

## 📞 Support

### Auteur
- **Nom** : Adam ELMERABET
- **Établissement** : EMSI - 4ème année IA et DATA
- **Email** : [votre email]
- **GitHub** : [votre profil]

### Signaler un Problème
1. Vérifier ce guide de dépannage
2. Consulter le code source commenté
3. Créer une issue sur le repository

---

**Date de création** : 27 Mars 2026  
**Version** : 1.0.0  
**Statut** : ✅ Production ready

---

*Bonne exploration de votre projet de Machine Learning !* 🚀
