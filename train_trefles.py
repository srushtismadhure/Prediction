import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pytensor.tensor as pt
import numpy as np, json
import json
import streamlit as st




matplotlib.use("TkAgg")

# 1. Load data
df = pd.read_csv("t2dm_trefles_synthetic_10000.csv")
print(df.head())

# 2. Exploratory Data Analysis
eda_df = df.copy()

# a. Add readable gender labels on the copy only
gender_map = {0: 'Female', 1 : "Male"}
eda_df["gender_label"] = eda_df["gender"].map(gender_map)



# -----------------------------
# Plot 1: Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
gender_counts = eda_df["gender_label"].value_counts()

plt.bar(gender_counts.index, gender_counts.values,
        color=['#4c72b0', '#FFC0CB'])

plt.title("Patient Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig("gender_plot.png")
plt.show()      # show first plot
plt.close()     # close so next plot is clean


# -----------------------------
# Plot 2: Age Distribution
# -----------------------------
plt.figure(figsize=(8,4))
plt.hist(eda_df["age"], bins=20,
         color="#84b6f4", edgecolor="black")

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig("age_plot.png")
plt.show()      # show second plot
plt.close()



print(eda_df['age'].head())
print(eda_df['age'].describe())




# -----------------------------
# Plot 3: ICD code distribution
# -----------------------------

# Define ICD prefixes (feature groups)
# Automatically detect ICD prefixes
icd_cols = [col for col in eda_df.columns if col.startswith("icd_")]

# Extract unique prefixes (everything before the last "_<number>")
import re

icd_prefixes = sorted(set([
    re.match(r"(icd_[a-zA-Z]+)", col).group(1)
    for col in icd_cols
]))

# Collect ICD columns for each prefix
icd_groups = {}
for prefix in icd_prefixes:
    cols = [c for c in eda_df.columns if c.startswith(prefix)]
    icd_groups[prefix] = cols

# Turn that into a summary table
icd_counts = pd.DataFrame({
    "icd_group": list(icd_groups.keys()),
    "n_columns": [len(cols) for cols in icd_groups.values()]
})

print(icd_counts)

# Bar chart: number of ICD features per group
plt.figure(figsize=(8, 4))
plt.bar(icd_counts["icd_group"], icd_counts["n_columns"])
plt.title("Number of ICD Feature Columns per Group")
plt.xlabel("ICD Group")
plt.ylabel("Number of Columns")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("icd_group_column_counts.png")
plt.show()
plt.close()


# -----------------------------
# Plot 4: Complication plot
# -----------------------------
# Detect columns that do NOT start with feature prefixes
feature_prefixes = ["icd_", "med_", "patient_id", "age", "gender"]

possible_targets = [
    col for col in eda_df.columns
    if not any(col.startswith(prefix) for prefix in feature_prefixes)
]

# Keep only binary columns (complications)
complication_cols = [
    col for col in possible_targets
    if set(df[col].unique()) <= {0, 1}
]


import seaborn as sns

plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=[1]*len(complication_cols),
    y=complication_cols,
    s=200, color="#FF6347"
)
plt.title("Complication Types Available in Dataset")
plt.xticks([])  # hide x-axis
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("complication_label_visualization.png")
plt.show()
plt.close()


# -----------------------------
# Plot 5: Medication Plot
# -----------------------------
med_cols = [col for col in eda_df.columns if col.startswith("med_")]
# Make pretty labels
pretty_meds = [col.replace("med_", "").replace("_", " ").title() for col in med_cols]

plt.figure(figsize=(6, 8))
plt.barh(pretty_meds, [1]*len(pretty_meds), color="#FFA07A")
plt.title("Medication Types in the Dataset")
plt.xlabel("Dummy Value (Not Meaningful)")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("medication_name_list.png")
plt.show()
plt.close()

# ----------------------------------------------
# 5. BUILD FEATURE LIST (X)
# ----------------------------------------------

# Drop non-numeric helper column
eda_df = eda_df.drop(columns=["gender_label"])

feature_cols = [
    col for col in eda_df.columns
    if col not in ["patient_id"] + complication_cols
]

print("Number of feature columns:", len(feature_cols))
print("First few feature cols:", feature_cols[:10])

# Ensure all feature columns are numeric
eda_df[feature_cols] = eda_df[feature_cols].apply(pd.to_numeric)

X = eda_df[feature_cols].astype(float).values      # (N, p)
Y = eda_df[complication_cols].astype(int).values  # (N, K)
N, p = X.shape
K = Y.shape[1]

print("N, p, K =", N, p, K)
print("Sample X row:", X[0])


# ----------------------------------------------
# 6. grouping them into larger groups
# ----------------------------------------------
group_ids = []
group_labels = []

for c in feature_cols:
    if c in ["age", "gender"]:
        group_ids.append(0); group_labels.append("demo")
    elif c.startswith("icd_cardio_"):
        group_ids.append(1); group_labels.append("icd_cardio")
    elif c.startswith("icd_infect_"):
        group_ids.append(2); group_labels.append("icd_infect")
    elif c.startswith("icd_electrolyte_"):
        group_ids.append(3); group_labels.append("icd_elect")
    elif c.startswith("icd_renal_"):
        group_ids.append(4); group_labels.append("icd_renal")
    elif c.startswith("icd_neuro_"):
        group_ids.append(5); group_labels.append("icd_neuro")
    elif c.startswith("icd_eye_"):
        group_ids.append(6); group_labels.append("icd_eye")
    elif c.startswith("icd_misc_"):
        group_ids.append(7); group_labels.append("icd_misc")
    else:
        # meds
        group_ids.append(8); group_labels.append("med")

group_ids = np.array(group_ids)
unique_groups = np.unique(group_ids)
print("Groups:", unique_groups)
for g in unique_groups:
    print(g, group_labels[group_ids.tolist().index(g)],
          "n_features =", np.sum(group_ids == g))


# store mean/std for later prediction
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0   # avoid division by zero
X_stdzd = (X - X_mean) / X_std


# prepare group index mapping: dict g -> indices of features in that group
groups = {}
for j, g in enumerate(group_ids):
    groups.setdefault(g, []).append(j)

with pm.Model() as trefles_model:

    # ---- 1. Task covariance (Ω) with LKJ prior ----
    # Cholesky factor for KxK covariance
    eta_omega = 2.0  # LKJ concentration parameter (moderate)
    chol_Omega, corr_Omega, sigma_Omega = pm.LKJCholeskyCov(
        "Omega_chol",
        n=K,
        eta=eta_omega,
        sd_dist=pm.Exponential.dist(1.0),
        compute_corr=True,
    )
    # cov_Omega = chol_Omega @ chol_Omega.T

    # ---- 2. Horseshoe-like global/local scales per group ----
    W_blocks = []
    for g, idxs in groups.items():
        idxs = np.array(idxs)
        Gz = len(idxs)

        # Group-specific feature covariance Σ_z via LKJ
        eta_sigma = 2.0
        chol_Sigma_z, corr_Sigma_z, sigma_Sigma_z = pm.LKJCholeskyCov(
            f"Sigma_chol_g{g}",
            n=Gz,
            eta=eta_sigma,
            sd_dist=pm.Exponential.dist(1.0),
            compute_corr=True,
        )

        # Global group scale τ_z (Half-Cauchy)
        tau_z = pm.HalfCauchy(f"tau_g{g}", beta=1.0)

        # Local scales λ_{j,k} (Half-Cauchy) for each coefficient in this group
        lam_z = pm.HalfCauchy(f"lambda_g{g}", beta=1.0, shape=(Gz, K))

        # Raw MatrixNormal block for group g
        W_raw_z = pm.MatrixNormal(
            f"W_raw_g{g}",
            mu=pt.zeros((Gz, K)),
            rowchol=chol_Sigma_z,
            colchol=chol_Omega,
            shape=(Gz, K)
        )

        # Apply horseshoe scaling: W_z = τ_z * λ_z * W_raw_z (elementwise)
        W_z = pm.Deterministic(
            f"W_g{g}",
            tau_z * lam_z * W_raw_z
        )

        W_blocks.append((idxs, W_z))

         # ---- 3. Assemble full W (p x K) from group blocks ----
    W_full = pt.zeros((p, K))
    for idxs, W_z in W_blocks:
        W_full = pt.set_subtensor(W_full[idxs, :], W_z)
    W_full = pm.Deterministic("W_full", W_full)

    # ---- 4. Data as shared variables ----
    X_shared = pm.Data("X_shared", X_stdzd)
    Y_shared = pm.Data("Y_shared", Y)

    # Linear predictor and likelihood
    eta = pm.Deterministic("eta", X_shared @ W_full)  # (N, K)
    # PyMC expects 1D/Broadcast for binary; we flatten tasks into vector:
    # Reshape N x K into (N*K,) and same for Y
    eta_flat = eta.reshape((-1,))
    Y_flat = Y_shared.reshape((-1,))

    pm.Bernoulli("Y_obs", logit_p=eta_flat, observed=Y_flat)

    # Ready to sample
with trefles_model:
    trace = pm.sample(
        draws=1000,          # posterior samples
        tune=1000,           # warmup
        chains=2,
        target_accept=0.9,
        random_seed=42
    )
    
az.summary(trace, var_names=["W_full", "Omega_chol"])


W_samples = trace.posterior["W_full"].values  # (chains, draws, p, K)
W_samples = W_samples.reshape(-1, p, K)
W_mean = W_samples.mean(axis=0)

logits = X_stdzd @ W_mean
probs  = 1 / (1 + np.exp(-logits))

risk_scores = pd.DataFrame(probs, columns=complication_cols)
risk_scores["patient_id"] = df["patient_id"]
print(risk_scores.head())



# Save the deployable model weights + preprocessing parameters
np.save("W_mean.npy", W_mean)
np.save("X_mean.npy", X_mean)
np.save("X_std.npy",  X_std)

with open("feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

with open("complication_cols.json", "w") as f:
    json.dump(complication_cols, f)

print("Saved model artifacts:")
print(" - W_mean.npy")
print(" - X_mean.npy")
print(" - X_std.npy")
print(" - feature_cols.json")
print(" - complication_cols.json")

# predictor.py
import json
import numpy as np

COMP_COLS = ["RET","NEU","NEP","VAS","CEL","PYE","OST","REN","HHS","KET","SEP","SHK"]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class TreflesScorer:
    def __init__(self, model_dir="."):
        self.W = np.load(f"{model_dir}/W_mean.npy")         # (p, K)
        self.X_mean = np.load(f"{model_dir}/X_mean.npy")    # (p,)
        self.X_std  = np.load(f"{model_dir}/X_std.npy")     # (p,)
        with open(f"{model_dir}/feature_cols.json") as f:
            self.feature_cols = json.load(f)

        # Safety: avoid division by zero
        self.X_std[self.X_std == 0] = 1.0

    def build_x_vector(self, x_dict: dict) -> np.ndarray:
        """
        x_dict: {feature_name: value} where missing features assumed 0
        Returns x in correct feature order, shape (p,)
        """
        x = np.array([x_dict.get(f, 0) for f in self.feature_cols], dtype=float)
        return x

    def predict_proba(self, x_dict: dict) -> dict:
        x = self.build_x_vector(x_dict)
        x_std = (x - self.X_mean) / self.X_std
        logits = x_std @ self.W                 # (K,)
        probs = sigmoid(logits)
        return dict(zip(COMP_COLS, probs))


import json
import numpy as np

COMP_COLS = ["RET","NEU","NEP","VAS","CEL","PYE","OST","REN","HHS","KET","SEP","SHK"]

def sigmoid(z):
    return 1/(1+np.exp(-z))

class TreflesScorer:
    def __init__(self, model_dir="."):
        self.W = np.load(f"{model_dir}/W_mean.npy")          # (p, K)
        self.X_mean = np.load(f"{model_dir}/X_mean.npy")     # (p,)
        self.X_std  = np.load(f"{model_dir}/X_std.npy")      # (p,)
        with open(f"{model_dir}/feature_cols.json") as f:
            self.feature_cols = json.load(f)

        self.X_std[self.X_std == 0] = 1.0

    def predict_proba(self, x_dict: dict) -> dict:
        x = np.array([x_dict.get(f, 0) for f in self.feature_cols], dtype=float)
        x_std = (x - self.X_mean) / self.X_std
        logits = x_std @ self.W                              # (K,)
        probs = sigmoid(logits)
        return dict(zip(COMP_COLS, probs))


# Map user-friendly symptom/history selections to the ICD feature flags your model understands.

SYMPTOM_TO_ICD = {
    "UTI symptoms (burning/frequency/urgency)": ["icd_infect_1", "icd_infect_2"],
    "Skin infection / cellulitis symptoms": ["icd_infect_5", "icd_misc_10"],
    "Fever / chills (infection-like)": ["icd_infect_3"],
    "Numbness / tingling (neuropathy-like)": ["icd_neuro_1", "icd_neuro_3"],
    "Blurred vision / eye issues": ["icd_eye_1", "icd_eye_2"],
    "Swelling / kidney concerns": ["icd_renal_1", "icd_renal_4"],
    "Chest pain / known cardiovascular history": ["icd_cardio_1", "icd_cardio_6"],
    "Dehydration / electrolyte issues": ["icd_electrolyte_1", "icd_electrolyte_2"],
}

# Your meds must match your dataset column names exactly (med_*).
MED_OPTIONS = [
    "med_insulin",
    "med_metformin",
    "med_sulfonylureas",
    "med_ace_inhibitor",
    "med_beta_blocker",
    "med_statins",
    "med_antibiotic",
    "med_diuretic",
    "med_anticoagulant",
    "med_glp1_agonist",
    "med_sglt2_inhibitor",
    # Add the rest of your med_* columns here
]

def build_feature_dict(age: int, gender: str, selected_meds: list, selected_symptoms: list) -> dict:
    x = {}
    x["age"] = int(age)
    x["gender"] = 1 if gender.lower() == "male" else 0  # adjust if your convention differs

    for m in selected_meds:
        x[m] = 1

    for s in selected_symptoms:
        for icd_feat in SYMPTOM_TO_ICD.get(s, []):
            x[icd_feat] = 1

    return x


COMP_COLS = ["RET","NEU","NEP","VAS","CEL","PYE","OST","REN","HHS","KET","SEP","SHK"]

def sigmoid(z):
    return 1/(1+np.exp(-z))

class TreflesScorer:
    def __init__(self, model_dir="."):
        self.W = np.load(f"{model_dir}/W_mean.npy")          # (p, K)
        self.X_mean = np.load(f"{model_dir}/X_mean.npy")     # (p,)
        self.X_std  = np.load(f"{model_dir}/X_std.npy")      # (p,)
        with open(f"{model_dir}/feature_cols.json") as f:
            self.feature_cols = json.load(f)

        self.X_std[self.X_std == 0] = 1.0

    def predict_proba(self, x_dict: dict) -> dict:
        x = np.array([x_dict.get(f, 0) for f in self.feature_cols], dtype=float)
        x_std = (x - self.X_mean) / self.X_std
        logits = x_std @ self.W                              # (K,)
        probs = sigmoid(logits)
        return dict(zip(COMP_COLS, probs))


# mapping.py

# Symptom/history selections -> ICD proxy flags (must exist in feature_cols.json)
SYMPTOM_TO_ICD = {
    "UTI symptoms (burning/frequency/urgency)": ["icd_infect_1", "icd_infect_2"],
    "Skin infection / cellulitis symptoms": ["icd_infect_5", "icd_misc_10"],
    "Fever / chills (infection-like)": ["icd_infect_3"],
    "Numbness / tingling (neuropathy-like)": ["icd_neuro_1", "icd_neuro_3"],
    "Blurred vision / eye issues": ["icd_eye_1", "icd_eye_2"],
    "Swelling / kidney concerns": ["icd_renal_1", "icd_renal_4"],
    "Chest pain / known cardiovascular history": ["icd_cardio_1", "icd_cardio_6"],
    "Dehydration / electrolyte issues": ["icd_electrolyte_1", "icd_electrolyte_2"],
}

# Medication columns (match your CSV header exactly)
MED_OPTIONS = [
    "med_insulin",
    "med_metformin",
    "med_sulfonylureas",
    "med_ace_inhibitor",
    "med_beta_blocker",
    "med_statins",
    "med_thiazolidinediones",
    "med_dpp4_inhibitor",
    "med_sglt2_inhibitor",
    "med_glp1_agonist",
    "med_nsaid",
    "med_antibiotic",
    "med_anticoagulant",
    "med_diuretic",
    "med_opioid",
    "med_antidepressant",
    "med_ppi",
    "med_antihistamine",
    "med_other",
]

def build_feature_dict(age: int, gender: str, selected_meds: list, selected_symptoms: list) -> dict:
    """
    Returns sparse feature dict: keys are feature names, values are numeric.
    Missing features are assumed 0 by the scorer.
    """
    x = {
        "age": int(age),
        "gender": 1 if gender.lower() == "male" else 0,  # matches your dataset: Male=1, Female=0
    }

    # meds
    for m in selected_meds:
        x[m] = 1

    # symptoms -> ICD proxies
    for s in selected_symptoms:
        for icd_feat in SYMPTOM_TO_ICD.get(s, []):
            x[icd_feat] = 1

    return x

