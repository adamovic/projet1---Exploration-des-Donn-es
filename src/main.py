# ==============================
# MAIN PROJECT EXECUTION
# ==============================
# Mini-projet ML - Prétraitement des Données
# Auteur: Adam ELMERABET - EMSI 4ème année IA et DATA

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Ajouter le chemin src au Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("MINI-PROJET MACHINE LEARNING - PRÉTRAITEMENT DES DONNÉES")
print("=" * 80)
print("Auteur: Adam ELMERABET")
print("Établissement: EMSI - 4ème année Ingénieur, Filière IA et DATA")
print("Date: 27 Mars 2026")
print("=" * 80)

# ==============================
# IMPORT DES MODULES
# ==============================
try:
    from experiments import run_main_experiments
    from sparse_experiments import run_sparse_experiments
    from outlier_experiments import run_outlier_experiments
    from pipeline_complete import run_complete_pipeline
    from preprocessing import test_preprocessing_functions
    print("✅ Tous les modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# ==============================
# MENU PRINCIPAL
# ==============================
def show_menu():
    print("\n" + "=" * 60)
    print("MENU PRINCIPAL - MINI-PROJET PRÉTRAITEMENT")
    print("=" * 60)
    print("1. 📊 Expériences principales de scaling")
    print("2. 🌿 Cas des données creuses (sparse)")
    print("3. 🎯 Analyse des outliers")
    print("4. 🔄 Pipeline complet avec validation croisée")
    print("5. 🔧 Tests des fonctions de prétraitement")
    print("6. 🚀 Exécuter toutes les expériences")
    print("7. 📈 Générer les visualisations")
    print("0. 🚪 Quitter")
    print("=" * 60)

def run_all_experiments():
    """Exécuter toutes les expériences dans l'ordre"""
    print("\n🚀 LANCEMENT DE TOUTES LES EXPÉRIENCES...")
    print("=" * 60)
    
    experiments = [
        ("Fonctions de prétraitement", test_preprocessing_functions),
        ("Expériences principales", run_main_experiments),
        ("Données creuses", run_sparse_experiments),
        ("Analyse des outliers", run_outlier_experiments),
        ("Pipeline complet", run_complete_pipeline)
    ]
    
    for name, func in experiments:
        print(f"\n{'='*60}")
        print(f"EXÉCUTION: {name}")
        print('='*60)
        try:
            func()
            print(f"✅ {name} terminé avec succès")
        except Exception as e:
            print(f"❌ Erreur dans {name}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 TOUTES LES EXPÉRIENCES TERMINÉES!")
    print("=" * 60)

def generate_visualizations():
    """Générer toutes les visualisations du projet"""
    print("\n📈 GÉNÉRATION DES VISUALISATIONS...")
    
    # Créer le dossier figures s'il n'existe pas
    os.makedirs('../figures', exist_ok=True)
    
    try:
        # Importer matplotlib pour les graphiques
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
        
        # Générer des données pour visualisation
        X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                 n_informative=2, random_state=42, n_clusters_per_class=1)
        
        # Appliquer différents scalers
        scalers = {
            "Original": None,
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "MaxAbsScaler": MaxAbsScaler()
        }
        
        # Créer une figure comparative
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, scaler) in enumerate(scalers.items()):
            ax = axes[idx]
            
            if scaler:
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # Scatter plot coloré par classe
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.7)
            ax.set_title(f'{name}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
            
            # Ajouter les statistiques
            ax.text(0.02, 0.98, f'Mean: {np.mean(X_scaled, axis=0)}\nStd: {np.std(X_scaled, axis=0)}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Supprimer le dernier subplot vide
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig('../figures/scaling_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ Graphique comparatif sauvegardé: figures/scaling_comparison.png")
        
        # Fermer les figures pour libérer la mémoire
        plt.close('all')
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération des visualisations: {e}")

# ==============================
# FONCTIONS WRAPPER POUR LES MODULES
# ==============================
def run_main_experiments():
    """Wrapper pour les expériences principales"""
    print("\n📊 DÉMARRAGE DES EXPÉRIENCES PRINCIPALES...")
    os.system('python experiments.py')

def run_sparse_experiments():
    """Wrapper pour les expériences sparse"""
    print("\n🌿 DÉMARRAGE DES EXPÉRIENCES SPARSE...")
    os.system('python sparse_experiments.py')

def run_outlier_experiments():
    """Wrapper pour les expériences outliers"""
    print("\n🎯 DÉMARRAGE DES EXPÉRIENCES OUTLIERS...")
    os.system('python outlier_experiments.py')

def run_complete_pipeline():
    """Wrapper pour le pipeline complet"""
    print("\n🔄 DÉMARRAGE DU PIPELINE COMPLET...")
    os.system('python pipeline_complete.py')

def test_preprocessing_functions():
    """Wrapper pour les tests de prétraitement"""
    print("\n🔧 TEST DES FONCTIONS DE PRÉTRAITEMENT...")
    os.system('python preprocessing.py')

# ==============================
# BOUCLE PRINCIPALE
# ==============================
def main():
    """Fonction principale du programme"""
    
    # Vérifier que le dossier figures existe
    os.makedirs('../figures', exist_ok=True)
    
    while True:
        show_menu()
        
        try:
            choice = input("\nVotre choix (0-7): ").strip()
            
            if choice == '0':
                print("\n👋 Au revoir!")
                break
            elif choice == '1':
                run_main_experiments()
            elif choice == '2':
                run_sparse_experiments()
            elif choice == '3':
                run_outlier_experiments()
            elif choice == '4':
                run_complete_pipeline()
            elif choice == '5':
                test_preprocessing_functions()
            elif choice == '6':
                run_all_experiments()
            elif choice == '7':
                generate_visualizations()
            else:
                print("❌ Choix invalide. Veuillez choisir entre 0 et 7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interruption détectée. Au revoir!")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
        input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main()