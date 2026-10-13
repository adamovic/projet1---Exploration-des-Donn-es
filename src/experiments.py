# ==============================
# EXPERIMENTS SCRIPT
# ==============================
# Compare different scaling techniques using pipelines
# and cross-validation to evaluate their impact on model performance

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import warnings
warnings.filterwarnings('ignore')


# ==============================
# CONFIGURATION
# ==============================
RANDOM_STATE = 42
N_SAMPLES = 1000
CV_FOLDS = 5
MAX_ITER = 1000


# ==============================
# GENERATE DATASET
# ==============================
print("=" * 60)
print("PREPROCESSING TECHNIQUES COMPARISON")
print("=" * 60)
print(f"\nGenerating dataset...")
print(f"  - Samples: {N_SAMPLES}")
print(f"  - Features: 20")
print(f"  - Classes: 2")

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=RANDOM_STATE
)

print(f"\nDataset shape: X={X.shape}, y={y.shape}")
print(f"Data range: Min={X.min():.2f}, Max={X.max():.2f}")
print(f"Mean={X.mean():.2f}, Std={X.std():.2f}")


# ==============================
# DEFINE SCALING TECHNIQUES
# ==============================
scalers = {
    "❌ No Scaling (baseline)": None,
    "✓ StandardScaler": StandardScaler(),
    "✓ MinMaxScaler": MinMaxScaler(),
    "✓ RobustScaler": RobustScaler(),
    "✓ MaxAbsScaler": MaxAbsScaler()
}

print(f"\n{'='*60}")
print("APPLYING MODELS WITH DIFFERENT SCALING TECHNIQUES")
print(f"{'='*60}\n")


# ==============================
# RUN EXPERIMENTS
# ==============================
results = {}

for name, scaler in scalers.items():
    print(f"Testing: {name}")
    
    if scaler:
        pipe = Pipeline([
            ("scaler", scaler),
            ("model", LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE))
        ])
    else:
        pipe = Pipeline([
            ("model", LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE))
        ])
    
    # Cross-validation
    scores = cross_val_score(pipe, X, y, cv=CV_FOLDS, scoring='accuracy')
    results[name] = scores
    
    print(f"  Fold scores:   {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean accuracy: {scores.mean():.4f}")
    print(f"  Std deviation: {scores.std():.4f}")
    print()


# ==============================
# SUMMARY RESULTS
# ==============================
print(f"\n{'='*60}")
print("SUMMARY TABLE")
print(f"{'='*60}\n")

print(f"{'Method':<30} {'Mean Acc':<12} {'Std Dev':<12} {'Min-Max':<15}")
print("-" * 70)

for name, scores in results.items():
    method = name.replace("❌ ", "").replace("✓ ", "")
    mean_acc = scores.mean()
    std_dev = scores.std()
    min_max = f"[{scores.min():.3f} - {scores.max():.3f}]"
    print(f"{method:<30} {mean_acc:<12.4f} {std_dev:<12.4f} {min_max:<15}")


# ==============================
# RANKING
# ==============================
print(f"\n{'='*60}")
print("RANKING (by mean accuracy)")
print(f"{'='*60}\n")

sorted_results = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)

for rank, (name, scores) in enumerate(sorted_results, 1):
    method = name.replace("❌ ", "").replace("✓ ", "")
    improvement = (scores.mean() - list(results.values())[0].mean()) * 100
    arrow = "↑" if improvement > 0 else "→"
    print(f"{rank}. {method:<28} {scores.mean():.4f}  {arrow} {improvement:+.2f}%")


print(f"\n{'='*60}\n")