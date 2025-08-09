# Lab05 Solution - IRCTC Stock Price Dataset
# Covers A1–A7 from Lab05.pdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ---------- Functions ----------

def load_and_clean(path, sheet=0):
    """Load Excel, clean numeric columns, and return DataFrame."""
    df = pd.read_excel("Lab Session Data1.xlsx")
    df.columns = [c.strip() for c in df.columns]

    # Clean Chg%
    if 'Chg%' in df.columns:
        df['Chg%'] = df['Chg%'].astype(str).str.replace('%', '').str.replace(',', '').replace('nan', '0')
        df['Chg%'] = pd.to_numeric(df['Chg%'], errors='coerce').fillna(0.0)

    # Clean numeric columns
    for col in ['Price', 'Open', 'High', 'Low', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').replace('nan', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop missing Price
    df = df.dropna(subset=['Price']).reset_index(drop=True)
    return df

def split_features_target(df, target_col='Price', features=None, test_size=0.25, random_state=42):
    """Split into train/test sets."""
    if features is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric if c != target_col]
    X = df[features].fillna(0.0)
    y = df[target_col].fillna(0.0)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_reg(X_train, y_train):
    """Train Linear Regression."""
    return LinearRegression().fit(X_train, y_train)

def compute_metrics(y_true, y_pred):
    """Compute MSE, RMSE, MAPE, R²."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    denom = np.where(y_true == 0, 1e-8, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAPE%': mape, 'R2': r2}

def kmeans_fit(X, n_clusters=2, random_state=42):
    """Fit KMeans."""
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5).fit(X)

def evaluate_clustering(X, labels):
    """Compute clustering metrics."""
    n_clusters = len(np.unique(labels))
    res = {}
    if n_clusters > 1 and X.shape[0] > n_clusters:
        try: res['Silhouette'] = silhouette_score(X, labels)
        except: res['Silhouette'] = np.nan
        try: res['Calinski_Harabasz'] = calinski_harabasz_score(X, labels)
        except: res['Calinski_Harabasz'] = np.nan
        try: res['Davies_Bouldin'] = davies_bouldin_score(X, labels)
        except: res['Davies_Bouldin'] = np.nan
    else:
        res['Silhouette'] = np.nan
        res['Calinski_Harabasz'] = np.nan
        res['Davies_Bouldin'] = np.nan
    return res

def evaluate_k_range(X, k_min=2, k_max=8, random_state=42):
    """Evaluate KMeans for a range of k values."""
    ks, inertias, silhouettes, ch_scores, db_scores = [], [], [], [], []
    for k in range(k_min, k_max+1):
        km = kmeans_fit(X, n_clusters=k, random_state=random_state)
        ks.append(k)
        inertias.append(km.inertia_)
        m = evaluate_clustering(X, km.labels_)
        silhouettes.append(m['Silhouette'])
        ch_scores.append(m['Calinski_Harabasz'])
        db_scores.append(m['Davies_Bouldin'])
    return {'k': ks, 'inertia': inertias, 'silhouette': silhouettes,
            'calinski_harabasz': ch_scores, 'davies_bouldin': db_scores}

def plot_simple(x, y, xlabel='k', ylabel='value', title='Plot'):
    """Simple line plot."""
    plt.figure(figsize=(8,4))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

# ---------- Main Execution ----------

FILEPATH = "Lab Session Data1.xlsx" 

# Load data
df = load_and_clean(FILEPATH)

# A1 & A2: One attribute (Open) → Price
features_one = ['Open']
X_train1, X_test1, y_train1, y_test1 = split_features_target(df, features=features_one)
reg1 = train_linear_reg(X_train1, y_train1)
metrics_train1 = compute_metrics(y_train1, reg1.predict(X_train1))
metrics_test1 = compute_metrics(y_test1, reg1.predict(X_test1))

# A3: All numeric attributes
numeric_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'Price']
X_train_all, X_test_all, y_train_all, y_test_all = split_features_target(df, features=numeric_feats)
reg_all = train_linear_reg(X_train_all, y_train_all)
metrics_train_all = compute_metrics(y_train_all, reg_all.predict(X_train_all))
metrics_test_all = compute_metrics(y_test_all, reg_all.predict(X_test_all))

# Show metrics
metrics_summary = pd.DataFrame([
    {'Model': 'LinearReg_Open', 'Dataset': 'Train', **metrics_train1},
    {'Model': 'LinearReg_Open', 'Dataset': 'Test', **metrics_test1},
    {'Model': 'LinearReg_All', 'Dataset': 'Train', **metrics_train_all},
    {'Model': 'LinearReg_All', 'Dataset': 'Test', **metrics_test_all},
])
print("\n=== Linear Regression Metrics Summary ===")
print(metrics_summary)

# A4–A7: KMeans clustering
X_cluster = df[numeric_feats].fillna(0.0)
k_res = evaluate_k_range(X_cluster, k_min=2, k_max=8)

# Plots
plot_simple(k_res['k'], k_res['inertia'], xlabel='k', ylabel='Inertia', title='Elbow Plot (Inertia vs k)')
plot_simple(k_res['k'], k_res['silhouette'], xlabel='k', ylabel='Silhouette Score', title='Silhouette Score vs k')
plot_simple(k_res['k'], k_res['calinski_harabasz'], xlabel='k', ylabel='Calinski-Harabasz Score', title='Calinski-Harabasz Score vs k')
plot_simple(k_res['k'], k_res['davies_bouldin'], xlabel='k', ylabel='Davies-Bouldin Index', title='Davies-Bouldin Index vs k')

# Clustering metrics table
clust_df = pd.DataFrame({
    'k': k_res['k'],
    'Inertia': k_res['inertia'],
    'Silhouette': k_res['silhouette'],
    'Calinski_Harabasz': k_res['calinski_harabasz'],
    'Davies_Bouldin': k_res['davies_bouldin']
})
print("\n=== Clustering Metrics ===")
print(clust_df)

