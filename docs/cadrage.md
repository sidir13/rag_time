# Document de cadrage — Projet RAG-Time
> **Projet M2 IA — RAG-Time**  
> Version 1.0 — Mai 2026  
> Équipe : Projet académique — M2 Intelligence Artificielle

---

## 1. Contexte et problématique

### 1.1 — Entreprise cible : LogiStore (fictive)

LogiStore est une société en croissance opérant dans la distribution et le service après-vente. Son système d'information repose sur des progiciels distincts et non connectés :

| Outil | Fonction |
|-------|----------|
| **ERP** (ex. SAP, Odoo) | Gestion commerciale, achats, ventes, livraisons |
| **CRM** (ex. Salesforce, HubSpot) | Suivi client, interactions, opportunités |
| **PIM** (ex. Akeneo) | Catalogue produits, fiches techniques |
| **Ticketing** (ex. Zendesk, Jira) | Support utilisateur, SAV |
| **GED / Collaboratif** (ex. SharePoint, Confluence) | Procédures internes, guides, contrats |

Cette fragmentation provoque trois problèmes opérationnels majeurs :
1. **Temps de recherche élevé** — les agents support cherchent manuellement dans plusieurs outils.
2. **Non-réutilisation des connaissances** — les résolutions de tickets passés ne sont pas exploitées.
3. **Perte de contexte** — aucune vision unifiée client/produit/historique au moment de la résolution.

### 1.2 — Problématique

> Comment permettre aux équipes métier et support de **retrouver rapidement des informations pertinentes** à travers les données fragmentées du SI, en s'appuyant sur l'intelligence artificielle générative ?

---

## 2. Objectif et vision cible

### 2.1 — Vision à court terme : MVP tickets de support

Développer un **moteur de recherche intelligent** sur les tickets de support SAV, permettant de :
- Retrouver des tickets similaires par thématique, client, produit, catégorie ou mots-clés métier.
- Présenter des résultats classés par pertinence (ranking explicable).
- Optionnellement synthétiser une réponse à partir des tickets les mieux classés (via LLM).

### 2.2 — Vision à moyen terme : RAG d'entreprise étendu

Étendre la même architecture à l'ensemble du SI de LogiStore :
- Données **ERP** : commandes, ventes, achats, livraisons.
- Données **CRM** : clients, interactions, segmentation.
- Données **PIM** : produits, familles, attributs techniques.
- Documents **GED** : procédures, contrats, FAQ, guides techniques.
- Historiques **SAV** : résolutions, escalades, SLA.

---

## 3. Périmètre du MVP

### 3.1 — Inclus dans le MVP

| Fonctionnalité | Détail |
|----------------|--------|
| **Ingestion CSV** | Dataset Kaggle Customer Support Tickets (200k tickets, 30 colonnes, 6 langues) |
| **Indexation sémantique** | Embeddings via HuggingFace Inference API (paraphrase-multilingual-MiniLM-L12-v2, 384 dims) |
| **Stockage vectoriel** | FAISS IndexFlatIP (recherche vectorielle exacte, persistante sur disque) |
| **Filtrage métadonnées** | 13 champs : produit, catégorie, priorité, statut, canal, région, langue, OS, abonnement, segment, escalade, SLA, dates |
| **Recherche multi-axe** | Par thématique, client, solution, produit, incident, catégorie, historique de résolution |
| **Génération LLM** | Via OpenRouter (gpt-4o-mini par défaut, configurable) |
| **Mode dégradé** | Retrieval-only si clé API absente |
| **Frontend** | Interface de chat avec filtres, affichage des sources et scores |
| **API REST FastAPI** | Endpoints `/api/chat`, `/api/health`, `/api/info` + frontend HTML statique |
| **Évaluation** | LLM-as-judge (faithfulness, relevance, precision, recall, hallucination, completeness) |

### 3.2 — Hors périmètre MVP (prévu en V2)

| Fonctionnalité | Justification |
|----------------|---------------|
| **Recherche hybride BM25 + vectorielle** | Nécessite OpenSearch ou Qdrant — non déployé dans cette phase |
| **RBAC / contrôle d'accès** | Requis pour la mise en production, hors scope académique |
| **Ingestion temps réel (webhooks)** | Prévu pour l'industrialisation V2 |
| **Connexion ERP/CRM/PIM** | Extension à moyen terme |
| **Reranking par cross-encoder** | Optimisation de la précision, prévu V2 |

---

## 4. Architecture retenue

L'architecture complète est détaillée dans [architecture_rag_v1.md](architecture_rag_v1.md).

### 4.1 — Vue synthétique du pipeline

```
[CSV Tickets] → [Ingestor] → [Chunking] → [Embedding HF API]
                                                    ↓
                                            [FAISS Index (disque)]
                                                    ↓
[Requête utilisateur] → [Embedding HF API] → [Recherche vectorielle]
                                                    ↓
                                            [Post-filtrage métadonnées]
                                                    ↓
                                            [Top-K chunks sélectionnés]
                                                    ↓
                                [Prompt construction] → [LLM OpenRouter]
                                                    ↓
                                            [Réponse + sources]
```

### 4.2 — Choix technologiques

| Composant | Technologie choisie | Justification |
|-----------|---------------------|---------------|
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (HF Inference API) | Multilingue 6 langues, zéro RAM local, plan gratuit HF |
| **Embeddings (alt.)** | `BAAI/bge-m3` (local) | Haute performance, 1024 dims, 8192 tokens context |
| **Vector store** | FAISS IndexFlatIP | Recherche exacte, ~1ms/requête, persistant, pas de dépendance C++ |
| **LLM** | OpenRouter (gpt-4o-mini) | API unifiée, accès multi-modèles, coûts contrôlés |
| **Frontend** | HTML statique | Déploiement rapide, filtres métadonnées intégrés |
| **Backend API** | FastAPI + Uvicorn | Async, léger, documentation automatique |
| **Recherche hybride (V2)** | OpenSearch / Qdrant | BM25 + vectoriel + filtres natifs dans un seul moteur |

---

## 5. Axes de recherche couverts par le MVP

Conformément au cahier des charges, la recherche dans les tickets est possible selon :

- **Par thématique** — recherche sémantique sur `issue_description` + `resolution_notes`
- **Par client** — filtre `customer_segment`
- **Par solution** — recherche sémantique sur `resolution_notes`
- **Par produit** — filtre `product`
- **Par incident** — recherche sémantique + filtre `category`
- **Par catégorie** — filtre `category`
- **Par historique de résolution** — filtre par plage de dates (`ticket_created_date`, `ticket_resolved_date`)

---

## 6. Trajectoire d'extension

### 6.1 — Extension à d'autres sources du SI

| Source | Type de données | Mécanisme d'ingestion | Index |
|--------|-----------------|-----------------------|-------|
| **ERP** | Structuré (JSON/CSV) | Export périodique + pipeline Python | FAISS / OpenSearch |
| **CRM** | Structuré (JSON/API REST) | Webhook ou sync quotidien | FAISS / OpenSearch |
| **PIM** | Structuré + semi-structuré | Export Akeneo (API) | FAISS / OpenSearch |
| **GED** | Documents (PDF, DOCX, MD) | Parsing + chunking par paragraphes | FAISS / OpenSearch |
| **SAV** | Historiques (CSV/API) | Export périodique | FAISS / OpenSearch |

### 6.2 — Feuille de route

```
Phase 1 (MVP actuel)   : Tickets de support — FAISS  + API REST
Phase 2 (V2)           : Hybride BM25 + vectoriel (OpenSearch), RBAC, webhooks
Phase 3 (V3)           : Extension ERP/CRM/PIM, datalake unifié, multi-tenant
Phase 4 (Production)   : Monitoring, observabilité, CI/CD, déploiement cloud
```

---

## 7. Stratégie d'évaluation

L'évaluation du système RAG est automatisée via `evaluate_rag.py` et `src/evaluate.py`.

### 7.1 — Métriques de retrieval (sans LLM)

| Métrique | Description |
|----------|-------------|
| **Hit Rate @ K** | Le ticket de référence est-il dans le top-K ? |
| **MRR @ K** | Mean Reciprocal Rank |
| **Precision @ K** | Chunks pertinents / K récupérés |
| **NDCG @ K** | Normalized Discounted Cumulative Gain |
| **Mean Similarity** | Score cosinus moyen des résultats |

### 7.2 — LLM-as-judge (via OpenRouter)

| Dimension | Description |
|-----------|-------------|
| **Faithfulness** | La réponse est-elle fidèle au contexte récupéré ? |
| **Answer Relevance** | La réponse répond-elle à la question posée ? |
| **Context Precision** | Les chunks récupérés sont-ils pertinents ? |
| **Context Recall** | Le contexte couvre-t-il la réponse attendue ? |
| **Hallucination Score** | La réponse contient-elle des informations inventées ? |
| **Completeness** | La réponse est-elle complète ? |

### 7.3 — Résultats obtenus (multilingual test — 200 requêtes)

Résultats disponibles dans `evaluation/` :
- `multilingual_test_report.json` — rapport complet
- `multilingual_gpt4o_report.json` — évaluation avec GPT-4o comme juge

---

## 8. Stratégie d'industrialisation

### 8.1 — Scripts automatisés

| Script | Rôle |
|--------|------|
| `scripts/import_data.py` | Téléchargement du dataset Kaggle |
| `scripts/reindex.py` | Ré-indexation complète (supprime et reconstruit FAISS) |
| `scripts/rebuild_vectorstore.py` | Migration de modèle d'embedding (sans re-parser le CSV) |

### 8.2 — Pipeline d'actualisation des index

Pour une mise en production, le pipeline d'ingestion devrait être déclenché :
1. **En batch** : tâche cron quotidienne sur les nouveaux tickets (filtre par `ticket_created_date`).
2. **En temps réel** : webhook Zendesk/Jira → endpoint FastAPI → ingestion + indexation immédiate.
3. **En migration** : `scripts/rebuild_vectorstore.py` pour changer de modèle d'embedding.

### 8.3 — Séparation des composants

```
src/           → bibliothèque métier (ingest, embed, rag, evaluate)
api.py         → service REST (FastAPI) — déployable indépendamment
app.py         → frontend — déployable indépendamment
scripts/       → scripts d'administration (CLI, non exposés)
tests/         → suite de tests (isolation, pas de dépendance au runtime)
```

### 8.4 — Configuration

Toutes les variables sensibles sont externalisées dans `.env` (voir `.env.example`) :
- `OPENROUTER_API_KEY` — accès aux LLM
- `HF_API_KEY` — accès aux embeddings HuggingFace
- `EMBED_MODEL` — modèle d'embedding configurable sans modifier le code
- `LLM_MODEL` — modèle LLM configurable
- `TOP_K` — nombre de chunks récupérés

### 8.5 — Déploiement

**Local :**
```bash
# API FastAPI
uvicorn api:app --reload --port 8000
```

**Conteneurisé (Docker — V2) :**
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements_api.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 9. Étude de risque

### 9.1 — Risques techniques

| Risque | Impact | Mitigation |
|--------|--------|-----------|
| **Hallucinations LLM** | Réponses incorrectes affichées comme faits | Prompt strict (répondre uniquement depuis le contexte), mode retrieval-only disponible |
| **Dégradation qualité embeddings** | Recall faible, mauvais ranking | Évaluation automatisée continue, benchmark régulier |
| **Dérive du modèle d'embedding** | Incompatibilité index/requête après mise à jour HF | Version explicite du modèle, script de re-indexation |
| **Latence API HF** | Temps de réponse dégradé | Fallback vers modèle local (BGE-M3), mise en cache des embeddings fréquents |
| **Coût API OpenRouter** | Dépassement budget | Top-K limité, mode retrieval-only par défaut, monitoring des tokens |

### 9.2 — Risques métier

| Risque | Impact | Mitigation |
|--------|--------|-----------|
| **Surconfiance utilisateur** | Adoption de mauvaises résolutions | Affichage systématique des sources, score de similarité visible |
| **Données obsolètes** | Résultats pertinents sur des tickets anciens | Filtre temporel exposé dans l'interface |
| **Qualité variable des tickets** | Résolutions incomplètes ou mal rédigées | Prétraitement + normalisation unicode à l'ingestion |

### 9.3 — Risques de sécurité et gouvernance des données

| Risque | Impact | Mitigation |
|--------|--------|-----------|
| **Données PII dans les tickets** | Exposition de noms/emails clients via l'API | Suppression systématique des champs PII à l'ingestion (`customer_name`, `customer_email`) |
| **Clés API exposées** | Accès non autorisé aux LLM | Variables d'environnement uniquement, `.env` dans `.gitignore` |
| **Envoi de données sensibles à une API externe** | Non-conformité RGPD | Seuls les textes de tickets anonymisés sont envoyés aux API ; pas de PII |
| **Absence de RBAC** | Tout utilisateur voit tous les tickets | À implémenter en V2 (JWT + rôles) |

---

## 10. Livrables du projet

| Livrable | Fichier | Statut |
|----------|---------|--------|
| Note de veille technologique | `docs/veille_rag.md` | ✅ Livré |
| Document de cadrage | `docs/cadrage.md` | ✅ Livré (ce document) |
| Schéma d'architecture | `docs/architecture_rag_v1.md` | ✅ Livré |
| EDA du dataset | `docs/eda_rapport.md` | ✅ Livré |
| MVP fonctionnel  | `app.py` + `src/` | ✅ Livré |
| API REST | `api.py` + `static/` | ✅ Livré |
| Évaluation LLM-as-judge | `evaluate_rag.py` + `src/evaluate.py` | ✅ Livré |
| Résultats d'évaluation | `evaluation/` | ✅ Livré |
| Dépôt Git documenté | `README.md` | ✅ Livré |
