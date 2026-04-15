# Analyse exploratoire — Customer Support Tickets (200k)

> **Dataset** : `data/raw/customer_support_tickets_200k.csv`
> **Période** : 1er janvier 2022 → 31 décembre 2024
> **Script** : `src/stats/generate_stats.py`
> **Sortie** : `data/processed/`

---

## Résumé exécutif

| Indicateur | Valeur |
|-----------|--------|
| **Tickets totaux** | 200 000 |
| **Période couverte** | 2022-01-01 → 2024-12-31 (3 ans) |
| **Colonnes** | 30 |
| **Valeurs manquantes** | Aucune |
| **Taux d'escalade** | **50,2 %** |
| **Taux de rupture SLA** | **50,0 %** |
| **CSAT moyen** | 3,0 / 10 |
| **Temps de résolution médian** | 120 heures (~5 jours) |
| **Temps de 1ère réponse médian** | 36 heures |
| **Score de complexité moyen** | 5,5 / 10 |

> **Note analytique** : les distributions sont toutes quasi-uniformes entre les modalités (design synthétique du dataset). Cela signifie que le dataset couvre parfaitement chaque combinaison de catégorie/produit/région/langue, ce qui est idéal pour l'entraînement d'un système RAG sans biais de couverture.

---

## 1. Aperçu général du dataset

### 1.1 Structure et qualité des données

Le dataset comporte **30 colonnes** couvrant l'ensemble du cycle de vie d'un ticket de support :

| Groupe | Colonnes |
|--------|---------|
| **Identifiant** | `ticket_id` |
| **Client** | `customer_name`, `customer_email`, `customer_age`, `customer_gender`, `customer_tenure_months`, `previous_tickets`, `subscription_type`, `customer_segment` |
| **Ticket** | `product`, `category`, `issue_description`, `resolution_notes`, `priority`, `status`, `channel` |
| **Géographie / langue** | `region`, `language`, `preferred_contact_time` |
| **Technique** | `operating_system`, `browser`, `payment_method` |
| **Performance** | `first_response_time_hours`, `resolution_time_hours`, `sla_breached`, `escalated` |
| **Qualité** | `customer_satisfaction_score`, `issue_complexity_score` |
| **Temporel** | `ticket_created_date`, `ticket_resolved_date` |

**Qualité des données : excellente** — aucune valeur manquante sur l'ensemble des 200 000 lignes.

![Valeurs manquantes](data/processed/01_missing_values.png)

---

## 2. Distribution des tickets par catégorie clé

### 2.1 Catégories de tickets

Les 10 catégories sont parfaitement équilibrées (~20 000 tickets chacune), ce qui est délibéré dans ce dataset synthétique.

| Catégorie | Volume |
|-----------|--------|
| Feature Request | 20 169 |
| Subscription Cancellation | 20 096 |
| Performance Issue | 20 074 |
| Security Concern | 20 040 |
| Login Issue | 20 002 |
| Payment Problem | 19 997 |
| Bug Report | 19 981 |
| Refund Request | 19 900 |
| Data Sync Issue | 19 877 |
| Account Suspension | 19 864 |

![Distribution catégories](data/processed/02_00_dist_category.png)

### 2.2 Priorité

Parfaitement équilibrée entre les 4 niveaux (Low / Medium / High / Urgent ≈ 50k chacun).

![Distribution priorité](data/processed/02_01_dist_priority.png)

### 2.3 Statut

Équilibre entre les 5 statuts : Open, In Progress, Pending Customer, Resolved, Closed (~40k chacun).

> **Implication RAG** : 40 % des tickets sont dans un état terminal (Resolved/Closed) et constituent le corpus principal pour la capitalisation des résolutions. Les 60 % restants (Open, In Progress, Pending) peuvent servir à la détection de tickets similaires en cours.

![Distribution statut](data/processed/02_02_dist_status.png)

### 2.4 Canal de contact

Répartition équitable sur 5 canaux : Web Form, Chat, Phone, Social Media, Email (~40k chacun).

![Distribution canal](data/processed/02_03_dist_channel.png)

### 2.5 Produit

10 produits couverts (~20k tickets chacun) : Billing System, CRM Platform, E-commerce Store, Cloud Storage, Mobile App, Analytics Dashboard, Web Portal, Payment Gateway, Subscription Service, API Service.

![Distribution produit](data/processed/02_04_dist_product.png)

### 2.6 Région

6 régions mondiales, quasi-équilibrées (~33k chacune) : Africa, Asia, South America, Europe, North America, Australia.

![Distribution région](data/processed/02_05_dist_region.png)

### 2.7 Type d'abonnement et segment client

Free / Basic / Premium / Enterprise (~50k chacun). Segments : Individual (67 098), Corporate (66 751), Small Business (66 151).

![Distribution abonnement](data/processed/02_06_dist_subscription_type.png)
![Distribution segment](data/processed/02_07_dist_customer_segment.png)

### 2.8 Langue

6 langues couvertes : Japanese (33 620), English (33 553), French (33 264), German (33 212), Spanish (33 184), Chinese (33 167).

> **Implication RAG** : le système doit obligatoirement supporter au minimum 6 langues. Le modèle d'embedding retenu (BGE-M3) couvre 100+ langues. L'analyseur BM25 OpenSearch devra être configuré avec un analyzer `standard` ou `icu_analyzer` (plugin ICU) plutôt que `french` pour gérer la diversité linguistique.

![Distribution langue](data/processed/02_08_dist_language.png)

### 2.9 Système d'exploitation

5 OS représentés : Android (40 141), iOS (40 063), Linux (40 020), MacOS (39 916), Windows (39 860).

![Distribution OS](data/processed/02_09_dist_operating_system.png)

---

## 3. Évolution temporelle

### 3.1 Volume mensuel de tickets

Le dataset couvre 36 mois (2022–2024) avec un volume régulier (~5 500/mois), sans saisonnalité marquée — caractéristique d'un dataset synthétique, mais représentatif d'un flux de support stable.

![Tickets par mois](data/processed/03a_tickets_per_month.png)

### 3.2 Tickets par jour de la semaine

Légère tendance à plus de tickets en milieu de semaine (Mardi–Jeudi), sans écart significatif.

![Tickets par jour](data/processed/03b_tickets_per_weekday.png)

### 3.3 Évolution des escalades

Le taux d'escalade (~50 %) est stable dans le temps, sans aggravation ni amélioration détectée. Ce taux est anormalement élevé et signale soit un design synthétique, soit un service support sous-dimensionné.

![Escalades timeline](data/processed/03c_escalation_timeline.png)

---

## 4. Performances et SLA

### 4.1 Taux d'escalade et rupture SLA

| Métrique | Taux |
|---------|------|
| **Taux d'escalade** | 50,2 % |
| **Taux de rupture SLA** | 50,0 % |

Ces deux taux à ~50 % sont caractéristiques d'un dataset synthétique généré de façon uniforme. Dans un contexte réel, un taux d'escalade > 20 % et un SLA breach > 15 % seraient déjà des signaux d'alarme.

![Escalade et SLA](data/processed/04a_escalation_sla_pie.png)

### 4.2 Temps de résolution par priorité

| Priorité | Médiane (h) |
|---------|------------|
| Medium | 119,9 |
| High | 120,3 |
| Urgent | 120,5 |
| Low | 121,1 |

> **Observation critique** : les temps de résolution ne varient **pas significativement selon la priorité** — ce qui est irréaliste (dans un vrai service support, les tickets Urgent sont traités en quelques heures vs. plusieurs jours pour le Low). Cela confirme le caractère synthétique du dataset. Pour le système RAG, cela n'affecte pas la qualité du retrieval mais signifie que la priorité ne sera pas un signal discriminant pour le ranking.

![Résolution par priorité](data/processed/04b_resolution_time_by_priority.png)

### 4.3 Temps de première réponse par canal

| Canal | Médiane (h) |
|-------|------------|
| Email | 36,0 |
| Phone | 36,3 |
| Chat | 36,3 |
| Web Form | 36,4 |
| Social Media | 36,5 |

Même observation : distribution uniforme, sans discrimination réelle entre canaux.

![Première réponse par canal](data/processed/04c_first_response_by_channel.png)

### 4.4 Taux de rupture SLA par catégorie

Taux quasi-uniforme à ~50 % pour toutes les catégories.

![SLA par catégorie](data/processed/04d_sla_by_category.png)

---

## 5. Satisfaction client (CSAT)

### 5.1 Distribution globale

| Métrique | Valeur |
|---------|-------|
| **Moyenne** | 3,0 / 10 |
| **Médiane** | 3,0 / 10 |
| **Écart-type** | 1,41 |

Le score de satisfaction est très bas (3/10) et homogène. Dans un contexte réel, un CSAT < 5 sur 10 serait un signal critique.

![Distribution CSAT](data/processed/05a_csat_distribution.png)

### 5.2 CSAT par priorité et canal

Le CSAT ne varie pas selon la priorité ni le canal de contact (toujours ~3,0), ce qui confirme la génération synthétique uniforme.

![CSAT par priorité](data/processed/05b_csat_by_priority.png)
![CSAT par canal](data/processed/05c_csat_by_channel.png)

### 5.3 CSAT : escaladés vs non-escaladés

Distribution identique, sans corrélation entre escalade et satisfaction.

> **Implication pour l'évaluation RAG** : le champ `customer_satisfaction_score` ne peut pas servir de proxy de qualité de résolution (trop uniforme). En revanche, `resolution_notes` + `status = Resolved/Closed` constituent un signal fiable de résolution réussie.

![CSAT vs escalade](data/processed/05d_csat_vs_escalation.png)

---

## 6. Profil client

### 6.1 Âge

Distribution uniforme entre 18 et 75 ans, avec une moyenne à 46,5 ans.

![Age clients](data/processed/06a_customer_age.png)

### 6.2 Genre

Répartition tripartite équilibrée : Male (33,3 %), Female (33,3 %), Other (33,4 %).

![Genre](data/processed/06b_gender_distribution.png)

### 6.3 Satisfaction vs Ancienneté

Aucune corrélation entre l'ancienneté du client (moyenne : 30,4 mois) et son score de satisfaction. La pente de régression est quasi nulle.

> **Implication RAG** : `customer_tenure_months` peut néanmoins être utile comme métadonnée de filtrage (ex : filtrer les solutions adaptées aux clients entreprise avec >24 mois d'ancienneté = usage avancé du produit).

![CSAT vs ancienneté](data/processed/06c_csat_vs_tenure.png)

---

## 7. Complexité et corrélations

### 7.1 Matrice de corrélation

Les corrélations entre variables numériques sont toutes proches de 0, confirmant l'indépendance des variables dans ce dataset synthétique.

![Corrélations](data/processed/07a_correlation_matrix.png)

### 7.2 Complexité par catégorie

Score de complexité uniforme à ~5,5 pour toutes les catégories.

> **Implication RAG** : `issue_complexity_score` est trop uniforme pour discriminer les tickets. En production réelle, ce score (ou un équivalent calculé) pourrait guider le routing : tickets complexes → agentic RAG avec multi-step reasoning ; tickets simples → retrieval direct.

![Complexité par catégorie](data/processed/07b_complexity_by_category.png)

### 7.3 Temps de résolution vs Complexité

Même constat : pas de variation significative avec la complexité (design synthétique).

![Résolution vs complexité](data/processed/07c_resolution_vs_complexity.png)

---

## 8. Analyse textuelle

### 8.1 Descriptions des tickets (`issue_description`)

Les termes les plus fréquents dans les descriptions reflètent les problèmes typiques de support B2B SaaS : *payment, account, data, sync, error, failed, update, login, access, subscription...* Ces termes forment le vocabulaire central à capturer par les embeddings.

![Wordcloud descriptions](data/processed/08a_wordcloud_issues.png)

### 8.2 Notes de résolution (`resolution_notes`)

Le corpus de résolutions utilise un vocabulaire plus technique et processuel : *restarted, restored, reset, reconfigured, updated, verified, cleared, synchronized...* Ce sont les termes clés pour la capitalisation des résolutions.

> **Implication chunking** : les `issue_description` et `resolution_notes` devraient être concaténées dans le même chunk (séparées par une balise claire) pour que l'embedding capture à la fois le problème et sa résolution — maximisant le recall lors de la recherche par symptôme.

![Wordcloud résolutions](data/processed/08b_wordcloud_resolutions.png)

---

## 9. Heatmap Catégorie × Produit

Distribution parfaitement uniforme (~2 000 tickets par combinaison catégorie/produit), ce qui garantit une couverture exhaustive du corpus pour le RAG.

![Heatmap catégorie-produit](data/processed/09_heatmap_category_product.png)

---

## 10. Implications pour le système RAG-Time

### 10.1 Structure des chunks recommandée

Sur la base de l'analyse du dataset, le chunk optimal pour ce corpus est :

```
title: {product} — {category}
body:
  Issue: {issue_description}
  Resolution: {resolution_notes}
metadata:
  ticket_id, product, category, priority, status, channel,
  region, language, operating_system, subscription_type,
  customer_segment, escalated, sla_breached,
  ticket_created_date, ticket_resolved_date
```

**Taille moyenne d'un chunk** (titre + description + résolution) : ~80–120 tokens → pas besoin de segmentation multi-chunks pour ce dataset. Chaque ticket est un chunk autonome.

### 10.2 Champs de filtrage métadonnées (OpenSearch)

| Champ | Type OpenSearch | Usage filtrage |
|-------|----------------|----------------|
| `product` | keyword | Filtrer par produit concerné |
| `category` | keyword | Filtrer par catégorie de ticket |
| `priority` | keyword | Filtrer par niveau d'urgence |
| `status` | keyword | Filtrer sur Resolved/Closed uniquement |
| `channel` | keyword | Stats, pas de filtrage direct |
| `region` | keyword | Cloisonnement géographique |
| `language` | keyword | Filtrage par langue client |
| `operating_system` | keyword | Diagnostic technique |
| `subscription_type` | keyword | Segmentation client |
| `customer_segment` | keyword | Segmentation client |
| `escalated` | keyword | Filtrer les cas complexes |
| `sla_breached` | keyword | Analyse de performance |
| `ticket_created_date` | date | Filtres temporels (6 derniers mois) |

### 10.3 Considérations sur l'évaluation

Puisque le dataset est synthétique avec des distributions uniformes :
- Le **CSAT** (3,0/10 uniforme) n'est **pas utilisable** comme signal de qualité
- Le **statut** (`Resolved`/`Closed`) est le signal de résolution réussie le plus fiable
- La **présence de `resolution_notes`** non vide est un indicateur de ticket documenté
- La génération de **Q&A synthétiques** pour l'évaluation doit cibler spécifiquement les combinaisons (catégorie, produit, symptôme) pour tester la couverture sémantique

### 10.4 Exclusions recommandées (PII)

Les champs suivants contiennent des données personnelles et doivent être **exclus de l'indexation** ou anonymisés :
- `customer_name` → remplacer par un identifiant anonyme
- `customer_email` → supprimer
- `customer_age`, `customer_gender` → conserver comme métadonnées agrégées (segment) mais pas dans le texte du chunk

---

## Fichiers générés

| Fichier | Description |
|---------|-------------|
| `data/processed/stats_summary.json` | Toutes les statistiques clés en JSON |
| `data/processed/01_missing_values.png` | Valeurs manquantes |
| `data/processed/01b_dtypes.png` | Types de colonnes |
| `data/processed/02_0[0-9]_dist_*.png` | Distributions des variables catégorielles |
| `data/processed/03a_tickets_per_month.png` | Volume mensuel |
| `data/processed/03b_tickets_per_weekday.png` | Volume par jour |
| `data/processed/03c_escalation_timeline.png` | Escalades dans le temps |
| `data/processed/04a_escalation_sla_pie.png` | Taux escalade/SLA |
| `data/processed/04b_resolution_time_by_priority.png` | Temps résolution par priorité |
| `data/processed/04c_first_response_by_channel.png` | Première réponse par canal |
| `data/processed/04d_sla_by_category.png` | SLA breach par catégorie |
| `data/processed/05a_csat_distribution.png` | Distribution CSAT |
| `data/processed/05b_csat_by_priority.png` | CSAT par priorité |
| `data/processed/05c_csat_by_channel.png` | CSAT par canal |
| `data/processed/05d_csat_vs_escalation.png` | CSAT escaladés vs non |
| `data/processed/06a_customer_age.png` | Distribution âge |
| `data/processed/06b_gender_distribution.png` | Répartition genre |
| `data/processed/06c_csat_vs_tenure.png` | CSAT vs ancienneté |
| `data/processed/07a_correlation_matrix.png` | Matrice de corrélation |
| `data/processed/07b_complexity_by_category.png` | Complexité par catégorie |
| `data/processed/07c_resolution_vs_complexity.png` | Résolution vs complexité |
| `data/processed/08a_wordcloud_issues.png` | Nuage mots descriptions |
| `data/processed/08b_wordcloud_resolutions.png` | Nuage mots résolutions |
| `data/processed/09_heatmap_category_product.png` | Heatmap catégorie × produit |
