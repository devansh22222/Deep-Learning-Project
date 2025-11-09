!pip install --upgrade pip
!pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.2 joblib openpyxl tensorflow==2.15.0 --force-reinstall
# Cell 1 — install libraries and mount Drive
!pip install -q pandas scikit-learn imbalanced-learn openpyxl joblib tensorflow==2.19.1

from google.colab import drive
drive.mount('/content/drive')

ARCHIVE_DRIVE_PATH = "/content/drive/MyDrive/WebPage/archive (1)"  # or your extracted folder

# ✅ fixed Cell 2
import os, glob, zipfile

# your original intended path
ARCHIVE_DRIVE_PATH = "/content/drive/MyDrive/WebPage/archive (1)"  # base name without .zip
ZIP_PATH = ARCHIVE_DRIVE_PATH + ".zip"  # the actual zip file path

# 1. if the unzipped folder doesn't exist but zip does, extract it
if not os.path.exists(ARCHIVE_DRIVE_PATH):
    if os.path.exists(ZIP_PATH):
        print(f"Found zip file at {ZIP_PATH}, extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(ARCHIVE_DRIVE_PATH)
        print("✅ Extraction complete.")
    else:
        # try searching for similar names in Drive
        print("Zip not found, searching for similar paths...")
        candidates = glob.glob("/content/drive/MyDrive/**/archive*zip", recursive=True)
        if candidates:
            print("Found possible zip files:")
            for c in candidates:
                print(" -", c)
        raise FileNotFoundError(f"Neither folder nor zip found at {ARCHIVE_DRIVE_PATH}")

# 2. now verify and list contents
if os.path.isdir(ARCHIVE_DRIVE_PATH):
    print("✅ Using extracted folder:", ARCHIVE_DRIVE_PATH)
    files = sorted(os.listdir(ARCHIVE_DRIVE_PATH))
    print(f"Found {len(files)} files (showing first 50):")
    for i, f in enumerate(files[:50], 1):
        print(f"{i:03d}. {f}")
else:
    raise FileNotFoundError(f"{ARCHIVE_DRIVE_PATH} is not a directory after extraction.")
# Cell 3 — locate all spreadsheets inside the extracted archive
import glob, os

# Use the extracted folder from your previous cell
DATA_FOLDER = ARCHIVE_DRIVE_PATH

# look recursively for all .csv/.xlsx files
patterns = ["**/*.csv", "**/*.xlsx", "**/*.xls"]
files = []
for p in patterns:
    files.extend(glob.glob(os.path.join(DATA_FOLDER, p), recursive=True))
files = sorted(files)

print(f"Found {len(files)} dataset files (showing first 50):")
for i, f in enumerate(files[:50], 1):
    print(f"{i:03d}. {f}")

if not files:
    raise RuntimeError("No data files found. Check that your archive contained CSV/XLSX files.")
# Cell 4 — load & combine a manageable sample of CSV/XLSX dataframes
import pandas as pd
import numpy as np

label_keywords = ['label','class','attack','type','traffic','category']

# --- limit rows per file ---
def safe_read(path, nrows=100000):  # read first 100k rows from each file
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path, nrows=nrows)
        else:
            return pd.read_excel(path, engine='openpyxl', nrows=nrows)
    except Exception as e:
        print("Skipping unreadable file:", path, "Error:", e)
        return None

dfs = []

# --- limit number of files to load ---
MAX_FILES = 3    # start small; you can later increase to 5, 10, etc.
print(f"Loading up to {MAX_FILES} files (each up to 100k rows)...\n")

for p in files[:MAX_FILES]:
    df = safe_read(p)
    if df is None or df.shape[0] < 10 or df.shape[1] < 3:
        continue
    # strip spaces and rename label column to 'Label'
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if any(kw in c.lower() for kw in label_keywords):
            df = df.rename(columns={c: 'Label'})
            break
    dfs.append(df)
    print(f"Loaded: {os.path.basename(p)} | shape {df.shape}")

print(f"\n✅ Loaded {len(dfs)} dataframes.")
if not dfs:
    raise RuntimeError("No valid datasets loaded — try adjusting MAX_FILES or file paths.")

combined = pd.concat(dfs, ignore_index=True, sort=False)
print("✅ Combined shape:", combined.shape)
# Cell 5 — clean combined data and create LabelBinary
drop_candidates = [
    'Flow ID','FlowID','Flow Id','Timestamp','Time','StartTime',
    'Src IP','SrcIP','Dst IP','DstIP','Source IP','Destination IP',
    'Src Port','Dst Port'
]
combined = combined.drop(columns=[c for c in drop_candidates if c in combined.columns], errors='ignore')

# Create binary label column
if 'Label' in combined.columns:
    combined['Label'] = combined['Label'].astype(str)
    combined['LabelBinary'] = combined['Label'].apply(lambda v: 'BENIGN' if 'benign' in v.lower() else 'ATTACK')
else:
    combined['LabelBinary'] = 'UNKNOWN'

print("Label distribution:\n", combined['LabelBinary'].value_counts(dropna=False))

# Drop columns with >50 % missing or single unique value
missing = combined.isnull().mean()
combined = combined.drop(columns=missing[missing > 0.5].index, errors='ignore')
single_val = [c for c in combined.columns if combined[c].nunique(dropna=True) <= 1 and c != 'LabelBinary']
combined = combined.drop(columns=single_val, errors='ignore')

print("✅ Cleaned shape:", combined.shape)
# Cell 6 — prepare features and target
import numpy as np, pandas as pd

X = combined.drop(columns=[c for c in ['Label','LabelBinary'] if c in combined.columns], errors='ignore')
y = combined['LabelBinary']

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:   # force numeric conversion if needed
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

cat_cols = [c for c in X.columns if c not in num_cols]
MAX_CAT_UNIQUE = 50
to_onehot = [c for c in cat_cols if X[c].nunique(dropna=True) <= MAX_CAT_UNIQUE]

X_num = X[num_cols].copy()
X_dum = pd.get_dummies(X[to_onehot].astype(str), drop_first=True) if to_onehot else pd.DataFrame()
X_final = pd.concat([X_num, X_dum], axis=1)

# Replace infinities & fill missing
X_final = X_final.replace([np.inf, -np.inf], np.nan)
X_final = X_final.fillna(X_final.median())

print("✅ Final feature shape:", X_final.shape)
# Cell 7 — scale + incremental PCA + split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
import joblib, os

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Incremental PCA (uses less memory)
pca = IncrementalPCA(n_components=min(100, X_scaled.shape[1]))  # limit comps to 100 or features
X_pca = pca.fit_transform(X_scaled)
print("✅ PCA reduced shape:", X_pca.shape)

OUT_DIR = "/content/drive/MyDrive/IDS_Project_from_archive"
os.makedirs(OUT_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
joblib.dump(pca, os.path.join(OUT_DIR, "pca.pkl"))

mask_known = ~y.isin(['UNKNOWN','unknown'])
X_pca_known = X_pca[mask_known.values]
y_known = y[mask_known].apply(lambda v: 'BENIGN' if 'benign' in str(v).lower() else 'ATTACK').values

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_pca_known, y_known, test_size=0.10, stratify=y_known, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1111, stratify=y_train_val, random_state=42)

X_train_ae = X_train[y_train == 'BENIGN']
X_val_ae = X_val[y_val == 'BENIGN']
print("✅ AE train:", X_train_ae.shape, "| AE val:", X_val_ae.shape)
# Cell 8 — build and train autoencoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

input_dim = X_train_ae.shape[1]
def build_ae(input_dim):
    i = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(i)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    b = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(b)
    x = layers.Dense(64, activation='relu')(x)
    o = layers.Dense(input_dim, activation='sigmoid')(x)
    return models.Model(i, o)

ae = build_ae(input_dim)
ae.compile(optimizer='adam', loss='mse')
es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = ae.fit(
    X_train_ae, X_train_ae,
    validation_data=(X_val_ae, X_val_ae),
    epochs=80,
    batch_size=128,
    callbacks=[es],
    verbose=1
)

ae.save(os.path.join(OUT_DIR, "autoencoder.h5"))
print("✅ Model trained and saved.")
# Cell 10 — plots (self-contained)
import numpy as np, matplotlib.pyplot as plt

# recompute mse if needed
try:
    mse
except NameError:
    X_test_recon = ae.predict(X_test)
    mse = np.mean((X_test - X_test_recon)**2, axis=1)
    y_test_bin = (y_test == 'ATTACK').astype(int)

plt.figure(figsize=(8,4))
plt.hist(mse[y_test_bin==0], bins=100, alpha=0.6, label='Benign')
plt.hist(mse[y_test_bin==1], bins=100, alpha=0.6, label='Attack')
plt.legend(); plt.xlabel("Reconstruction Error"); plt.title("Error Distribution"); plt.show()

# ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test_bin, mse)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.show()
