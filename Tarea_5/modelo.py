import joblib, json
from pathlib import Path

#ARTIF_DIR = Path("artefactos_modelo")

# Modelo entrenado
linreg = joblib.load("model.pkl")

# Artefactos
with open("cluster_maps.json", "r", encoding="utf-8") as f:
    cluster_maps = json.load(f)

with open("high_card_cols.json", "r", encoding="utf-8") as f:
    high_card_cols = json.load(f)

with open("low_card_cols.json", "r", encoding="utf-8") as f:
    low_card_cols = json.load(f)

with open("cols_union.json", "r", encoding="utf-8") as f:
    cols_union = json.load(f)

print("✅ Modelo y artefactos cargados, listos para predecir")

import pandas as pd

def transformar_nuevo(df_new, cluster_maps, high_card_cols, low_card_cols, cols_union):
    Z = df_new.copy()

    # 1) mapear clusters
    for col in high_card_cols:
        newc = f"{col}_cluster"
        Z[newc] = Z[col].map(cluster_maps.get(col, {})).fillna(-1).astype(int).astype(str)

    # 2) dummies
    cluster_cols = [f"{c}_cluster" for c in high_card_cols]
    Z_cat = pd.get_dummies(Z[low_card_cols + cluster_cols], drop_first=True)

    # 3) alinear columnas
    Z_cat = Z_cat.reindex(columns=cols_union, fill_value=0)

    return Z_cat

# Ejemplo de ticket nuevo
# Ejemplo de ticket nuevo
df_new = pd.DataFrame([{
    "caller_id": "Caller 2403",
    "opened_by": "Opened by  8",
    "location": "Location 165",
    "category": "Category 40",
    "subcategory": "Subcategory 215",
    "u_symptom": "Symptom 471",
    "priority": "2 - High",
    "assignment_group": "Group 24",
    "assigned_to": "Resolver 89",
    "u_priority_confirmation": "true", 
    "contact_type": "Phone"
}])

X_nuevo = transformar_nuevo(df_new, cluster_maps, high_card_cols, low_card_cols, cols_union)
pred = linreg.predict(X_nuevo)
print("Duración predicha:", float(pred[0]), "horas")