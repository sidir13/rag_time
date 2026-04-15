"""
generate_stats.py
-----------------
Analyse exploratoire complète du jeu de données customer_support_tickets_200k.csv
Produit des graphiques PNG dans data/processed/ et un fichier stats_summary.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "customer_support_tickets_200k.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE_CAT = "Set2"
PALETTE_SEQ = "Blues_d"
FIG_DPI = 150

def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return str(path.relative_to(ROOT))

# ── Load data ──────────────────────────────────────────────────────────────────
print("Chargement du dataset...")
df = pd.read_csv(DATA_RAW, parse_dates=["ticket_created_date", "ticket_resolved_date"])
print(f"  {len(df):,} lignes  |  {df.shape[1]} colonnes")

# Computed columns
df["resolution_time_days"] = df["resolution_time_hours"] / 24
df["first_response_time_hours"] = pd.to_numeric(df["first_response_time_hours"], errors="coerce")
df["resolution_time_hours"] = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
df["customer_satisfaction_score"] = pd.to_numeric(df["customer_satisfaction_score"], errors="coerce")
df["issue_complexity_score"] = pd.to_numeric(df["issue_complexity_score"], errors="coerce")
df["customer_age"] = pd.to_numeric(df["customer_age"], errors="coerce")
df["month"] = df["ticket_created_date"].dt.to_period("M")
df["year_month"] = df["ticket_created_date"].dt.to_period("M").astype(str)
df["weekday"] = df["ticket_created_date"].dt.day_name()

saved_paths = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. APERÇU GÉNÉRAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Aperçu général...")

# 1a. Valeurs manquantes
fig, ax = plt.subplots(figsize=(10, 5))
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if len(missing):
    sns.barplot(x=missing.values, y=missing.index, ax=ax, palette="Reds_d")
    ax.set_title("Valeurs manquantes par colonne", fontweight="bold")
    ax.set_xlabel("Nombre de valeurs manquantes")
    saved_paths["missing_values"] = save(fig, "01_missing_values")
else:
    plt.close(fig)
    saved_paths["missing_values"] = None

# 1b. Répartition des types de données
dtypes_count = df.dtypes.astype(str).value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(dtypes_count.values, labels=dtypes_count.index, autopct="%1.0f%%",
       colors=sns.color_palette(PALETTE_CAT, len(dtypes_count)))
ax.set_title("Types de colonnes", fontweight="bold")
saved_paths["dtypes"] = save(fig, "01b_dtypes")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTION DES TICKETS — CATÉGORIES CLÉS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Distributions des variables catégorielles...")

cat_cols = {
    "category": "Catégorie de ticket",
    "priority": "Priorité",
    "status": "Statut",
    "channel": "Canal de contact",
    "product": "Produit concerné",
    "region": "Région",
    "subscription_type": "Type d'abonnement",
    "customer_segment": "Segment client",
    "language": "Langue",
    "operating_system": "Système d'exploitation",
}

for i, (col, title) in enumerate(cat_cols.items()):
    counts = df[col].value_counts()
    n = len(counts)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette=PALETTE_CAT)
    ax.set_title(f"Distribution — {title}", fontweight="bold")
    ax.set_xlabel("Nombre de tickets")
    for bar, val in zip(ax.patches, counts.values):
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    saved_paths[f"dist_{col}"] = save(fig, f"02_{i:02d}_dist_{col}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ÉVOLUTION TEMPORELLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Évolution temporelle...")

# 3a. Tickets par mois
monthly = df.groupby("year_month").size().reset_index(name="count")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly["year_month"], monthly["count"], marker="o", linewidth=2, color="#2563eb")
ax.fill_between(range(len(monthly)), monthly["count"], alpha=0.15, color="#2563eb")
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["year_month"], rotation=45, ha="right", fontsize=8)
ax.set_title("Volume de tickets par mois", fontweight="bold")
ax.set_ylabel("Nombre de tickets")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
saved_paths["tickets_per_month"] = save(fig, "03a_tickets_per_month")

# 3b. Tickets par jour de la semaine
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_counts = df["weekday"].value_counts().reindex(weekday_order)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=weekday_counts.index, y=weekday_counts.values, ax=ax, palette="Blues_d")
ax.set_title("Volume de tickets par jour de la semaine", fontweight="bold")
ax.set_ylabel("Nombre de tickets")
saved_paths["tickets_per_weekday"] = save(fig, "03b_tickets_per_weekday")

# 3c. Escalades par mois
esc_monthly = df[df["escalated"] == "Yes"].groupby("year_month").size().reset_index(name="escalated")
esc_monthly = esc_monthly.merge(monthly, on="year_month")
esc_monthly["escalation_rate"] = esc_monthly["escalated"] / esc_monthly["count"] * 100
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(esc_monthly["year_month"], esc_monthly["escalated"], color="#dc2626", alpha=0.7, label="Escaladés")
ax.bar(esc_monthly["year_month"], esc_monthly["count"] - esc_monthly["escalated"],
       bottom=esc_monthly["escalated"], color="#86efac", alpha=0.7, label="Non escaladés")
ax.set_xticks(range(len(esc_monthly)))
ax.set_xticklabels(esc_monthly["year_month"], rotation=45, ha="right", fontsize=8)
ax.set_title("Tickets escaladés vs non escaladés par mois", fontweight="bold")
ax.legend()
saved_paths["escalation_timeline"] = save(fig, "03c_escalation_timeline")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PERFORMANCES & SLA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Performances et SLA...")

# 4a. Taux d'escalade + taux SLA breached
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, label, color in zip(
    axes,
    ["escalated", "sla_breached"],
    ["Escaladés", "SLA non respecté"],
    ["#dc2626", "#f59e0b"]
):
    counts = df[col].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=[color, "#d1d5db"], startangle=90)
    ax.set_title(label, fontweight="bold")
fig.suptitle("Taux d'escalade et de rupture SLA", fontweight="bold", fontsize=13)
saved_paths["escalation_sla_pie"] = save(fig, "04a_escalation_sla_pie")

# 4b. Temps de résolution par priorité (boxplot)
fig, ax = plt.subplots(figsize=(10, 5))
priority_order = ["Low", "Medium", "High", "Urgent"]
data_for_box = [df[df["priority"] == p]["resolution_time_hours"].dropna() for p in priority_order]
bp = ax.boxplot(data_for_box, labels=priority_order, patch_artist=True, showfliers=False)
colors = ["#86efac", "#fde68a", "#f97316", "#dc2626"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_title("Temps de résolution (heures) par priorité", fontweight="bold")
ax.set_ylabel("Heures")
saved_paths["resolution_time_by_priority"] = save(fig, "04b_resolution_time_by_priority")

# 4c. Temps de première réponse par canal
fig, ax = plt.subplots(figsize=(10, 5))
channel_resp = df.groupby("channel")["first_response_time_hours"].median().sort_values()
sns.barplot(x=channel_resp.values, y=channel_resp.index, ax=ax, palette="Oranges_d")
ax.set_title("Temps de première réponse médian par canal (heures)", fontweight="bold")
ax.set_xlabel("Heures (médiane)")
saved_paths["first_response_by_channel"] = save(fig, "04c_first_response_by_channel")

# 4d. SLA breached par catégorie
sla_by_cat = df.groupby("category")["sla_breached"].apply(
    lambda x: (x == "Yes").sum() / len(x) * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=sla_by_cat.values, y=sla_by_cat.index, ax=ax, palette="Reds_d")
ax.set_title("Taux de rupture SLA par catégorie (%)", fontweight="bold")
ax.set_xlabel("% tickets SLA breached")
saved_paths["sla_by_category"] = save(fig, "04d_sla_by_category")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SATISFACTION CLIENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Satisfaction client...")

# 5a. Distribution du score de satisfaction
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["customer_satisfaction_score"].dropna(), bins=10, color="#2563eb", edgecolor="white", alpha=0.85)
ax.set_title("Distribution du score de satisfaction (CSAT)", fontweight="bold")
ax.set_xlabel("Score (1–10)")
ax.set_ylabel("Nombre de tickets")
med = df["customer_satisfaction_score"].median()
ax.axvline(med, color="red", linestyle="--", label=f"Médiane : {med:.1f}")
ax.legend()
saved_paths["csat_distribution"] = save(fig, "05a_csat_distribution")

# 5b. Satisfaction par priorité
fig, ax = plt.subplots(figsize=(9, 5))
priority_sat = df.groupby("priority")["customer_satisfaction_score"].mean().reindex(priority_order)
sns.barplot(x=priority_order, y=priority_sat.values, ax=ax, palette=colors)
ax.set_title("Score de satisfaction moyen par priorité", fontweight="bold")
ax.set_ylabel("CSAT moyen")
ax.set_ylim(0, 10)
for bar, val in zip(ax.patches, priority_sat.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.2f}",
            ha="center", fontsize=10, fontweight="bold")
saved_paths["csat_by_priority"] = save(fig, "05b_csat_by_priority")

# 5c. Satisfaction par canal
fig, ax = plt.subplots(figsize=(10, 5))
sat_by_channel = df.groupby("channel")["customer_satisfaction_score"].mean().sort_values()
sns.barplot(x=sat_by_channel.values, y=sat_by_channel.index, ax=ax, palette="Blues_d")
ax.set_title("Score de satisfaction moyen par canal", fontweight="bold")
ax.set_xlabel("CSAT moyen")
ax.set_xlim(0, 10)
saved_paths["csat_by_channel"] = save(fig, "05c_csat_by_channel")

# 5d. CSAT escaladés vs non escaladés (histogramme superposé)
fig, ax = plt.subplots(figsize=(7, 5))
for label, color in [("No", "#2563eb"), ("Yes", "#dc2626")]:
    vals = df[df["escalated"] == label]["customer_satisfaction_score"].dropna()
    ax.hist(vals, bins=10, alpha=0.55, color=color,
            label=f"{'Non escaladé' if label == 'No' else 'Escaladé'}", density=True)
ax.set_title("Distribution CSAT : escaladés vs non escaladés", fontweight="bold")
ax.set_xlabel("Score de satisfaction")
ax.set_ylabel("Densité")
ax.legend()
saved_paths["csat_vs_escalation"] = save(fig, "05d_csat_vs_escalation")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PROFIL CLIENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Profil client...")

# 6a. Distribution des âges
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["customer_age"].dropna(), bins=30, color="#7c3aed", edgecolor="white", alpha=0.85)
ax.set_title("Distribution de l'âge des clients", fontweight="bold")
ax.set_xlabel("Âge")
ax.set_ylabel("Nombre de clients")
saved_paths["customer_age"] = save(fig, "06a_customer_age")

# 6b. Tickets par genre
fig, ax = plt.subplots(figsize=(7, 5))
gender_counts = df["customer_gender"].value_counts()
ax.pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%",
       colors=sns.color_palette("Set2", len(gender_counts)))
ax.set_title("Répartition par genre", fontweight="bold")
saved_paths["gender_distribution"] = save(fig, "06b_gender_distribution")

# 6c. Satisfaction vs ancienneté (scatter + regression line)
fig, ax = plt.subplots(figsize=(9, 5))
sample = df[["customer_tenure_months", "customer_satisfaction_score"]].dropna().sample(min(5000, len(df)), random_state=42)
ax.scatter(sample["customer_tenure_months"], sample["customer_satisfaction_score"],
           alpha=0.15, s=10, color="#2563eb")
m, b = np.polyfit(sample["customer_tenure_months"], sample["customer_satisfaction_score"], 1)
x_line = np.linspace(sample["customer_tenure_months"].min(), sample["customer_tenure_months"].max(), 100)
ax.plot(x_line, m * x_line + b, color="red", linewidth=2, label=f"Tendance (slope={m:.4f})")
ax.set_title("Satisfaction vs Ancienneté client (mois)", fontweight="bold")
ax.set_xlabel("Ancienneté (mois)")
ax.set_ylabel("Score de satisfaction")
ax.legend()
saved_paths["csat_vs_tenure"] = save(fig, "06c_csat_vs_tenure")

# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPLEXITÉ & CORRÉLATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Complexité et corrélations...")

# 7a. Heatmap de corrélation
num_cols = ["customer_age", "customer_tenure_months", "previous_tickets",
            "customer_satisfaction_score", "first_response_time_hours",
            "resolution_time_hours", "issue_complexity_score"]
corr = df[num_cols].dropna().corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask,
            ax=ax, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title("Matrice de corrélation — variables numériques", fontweight="bold")
saved_paths["correlation_matrix"] = save(fig, "07a_correlation_matrix")

# 7b. Complexité par catégorie
fig, ax = plt.subplots(figsize=(10, 6))
cat_complex = df.groupby("category")["issue_complexity_score"].mean().sort_values(ascending=False)
sns.barplot(x=cat_complex.values, y=cat_complex.index, ax=ax, palette="Purples_d")
ax.set_title("Score de complexité moyen par catégorie", fontweight="bold")
ax.set_xlabel("Complexité moyenne (1–10)")
saved_paths["complexity_by_category"] = save(fig, "07b_complexity_by_category")

# 7c. Temps résolution vs complexité (boxplot par quintile)
df["complexity_quintile"] = pd.qcut(df["issue_complexity_score"], q=5,
                                     labels=["Q1\n(faible)", "Q2", "Q3", "Q4", "Q5\n(élevé)"])
fig, ax = plt.subplots(figsize=(10, 5))
groups = [df[df["complexity_quintile"] == q]["resolution_time_hours"].dropna()
          for q in ["Q1\n(faible)", "Q2", "Q3", "Q4", "Q5\n(élevé)"]]
bp = ax.boxplot(groups, labels=["Q1\n(faible)", "Q2", "Q3", "Q4", "Q5\n(élevé)"],
                patch_artist=True, showfliers=False)
for patch in bp["boxes"]:
    patch.set_facecolor("#c4b5fd")
ax.set_title("Temps de résolution (h) vs Complexité (quintiles)", fontweight="bold")
ax.set_ylabel("Heures")
saved_paths["resolution_vs_complexity"] = save(fig, "07c_resolution_vs_complexity")

# ══════════════════════════════════════════════════════════════════════════════
# 8. WORD CLOUDS — DESCRIPTIONS & RÉSOLUTIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Word clouds...")

def make_wordcloud(series, title, fname):
    text = " ".join(series.dropna().str.lower().tolist())
    wc = WordCloud(width=1200, height=500, background_color="white",
                   colormap="Blues", max_words=100,
                   stopwords={"the", "a", "an", "is", "was", "to", "of", "and",
                               "in", "for", "my", "i", "it", "on", "at", "with",
                               "that", "this", "has", "have", "be", "not", "but",
                               "are", "from", "by", "after", "been", "or", "le",
                               "la", "les", "et", "du", "de", "un", "une", "des",
                               "en", "que", "qui", "ne", "pas"}).generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=14)
    return save(fig, fname)

saved_paths["wordcloud_issues"] = make_wordcloud(
    df["issue_description"].sample(min(20000, len(df)), random_state=42),
    "Nuage de mots — Descriptions des tickets",
    "08a_wordcloud_issues"
)
saved_paths["wordcloud_resolutions"] = make_wordcloud(
    df["resolution_notes"].sample(min(20000, len(df)), random_state=42),
    "Nuage de mots — Notes de résolution",
    "08b_wordcloud_resolutions"
)

# ══════════════════════════════════════════════════════════════════════════════
# 9. HEATMAP CATÉGORIE × PRODUIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Heatmap catégorie × produit...")

pivot = df.groupby(["category", "product"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.3)
ax.set_title("Volume de tickets — Catégorie × Produit", fontweight="bold")
ax.set_xlabel("Produit")
ax.set_ylabel("Catégorie")
plt.xticks(rotation=45, ha="right")
saved_paths["heatmap_category_product"] = save(fig, "09_heatmap_category_product")

# ══════════════════════════════════════════════════════════════════════════════
# 10. STATS RÉSUMÉ JSON
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Calcul des statistiques clés...")

stats = {
    "total_tickets": int(len(df)),
    "date_range": {
        "start": str(df["ticket_created_date"].min().date()),
        "end": str(df["ticket_created_date"].max().date()),
    },
    "columns": list(df.columns),
    "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().any()},
    "tickets_by_status": df["status"].value_counts().to_dict(),
    "tickets_by_priority": df["priority"].value_counts().to_dict(),
    "tickets_by_category": df["category"].value_counts().to_dict(),
    "tickets_by_product": df["product"].value_counts().to_dict(),
    "tickets_by_channel": df["channel"].value_counts().to_dict(),
    "tickets_by_region": df["region"].value_counts().to_dict(),
    "tickets_by_language": df["language"].value_counts().to_dict(),
    "tickets_by_subscription": df["subscription_type"].value_counts().to_dict(),
    "tickets_by_segment": df["customer_segment"].value_counts().to_dict(),
    "escalation_rate_pct": round((df["escalated"] == "Yes").mean() * 100, 2),
    "sla_breach_rate_pct": round((df["sla_breached"] == "Yes").mean() * 100, 2),
    "csat": {
        "mean": round(df["customer_satisfaction_score"].mean(), 2),
        "median": round(df["customer_satisfaction_score"].median(), 2),
        "std": round(df["customer_satisfaction_score"].std(), 2),
        "by_priority": df.groupby("priority")["customer_satisfaction_score"].mean().round(2).to_dict(),
        "by_channel": df.groupby("channel")["customer_satisfaction_score"].mean().round(2).to_dict(),
    },
    "resolution_time_hours": {
        "mean": round(df["resolution_time_hours"].mean(), 2),
        "median": round(df["resolution_time_hours"].median(), 2),
        "p75": round(df["resolution_time_hours"].quantile(0.75), 2),
        "p95": round(df["resolution_time_hours"].quantile(0.95), 2),
        "by_priority": df.groupby("priority")["resolution_time_hours"].median().round(2).to_dict(),
    },
    "first_response_time_hours": {
        "mean": round(df["first_response_time_hours"].mean(), 2),
        "median": round(df["first_response_time_hours"].median(), 2),
        "by_channel": df.groupby("channel")["first_response_time_hours"].median().round(2).to_dict(),
    },
    "issue_complexity_score": {
        "mean": round(df["issue_complexity_score"].mean(), 2),
        "median": round(df["issue_complexity_score"].median(), 2),
        "by_category": df.groupby("category")["issue_complexity_score"].mean().round(2).to_dict(),
    },
    "customer_age": {
        "mean": round(df["customer_age"].mean(), 1),
        "min": int(df["customer_age"].min()),
        "max": int(df["customer_age"].max()),
    },
    "customer_tenure_months": {
        "mean": round(df["customer_tenure_months"].mean(), 1),
        "median": float(df["customer_tenure_months"].median()),
    },
    "customer_gender": df["customer_gender"].value_counts().to_dict(),
    "operating_systems": df["operating_system"].value_counts().to_dict(),
    "saved_plots": saved_paths,
}

stats_path = OUT_DIR / "stats_summary.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"  [OK] {stats_path.name}")

print("\n=== TERMINE ===")
print(f"Graphiques generes dans : {OUT_DIR}")
print(f"Statistiques JSON       : {stats_path}")
