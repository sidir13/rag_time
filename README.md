# RAG-Time

Moteur de recherche RAG (Retrieval-Augmented Generation) sur tickets de support client.  
Projet M2 IA — Mai 2026.

---

## Vue d'ensemble

RAG-Time est un MVP de moteur de recherche d'entreprise intelligent, centré sur les tickets de support SAV. Il permet de retrouver des tickets similaires par thématique, produit, catégorie ou mots-clés métier, et peut synthétiser une réponse via un LLM (OpenRouter).

**Dataset** : [Customer Support Tickets Dataset (200k+)](https://www.kaggle.com/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records) — 30 colonnes, 6 langues, 10 produits, 3 ans.

```
graph TB
    subgraph SOURCES[" Source de données"]
        TICKET["Tickets CSV (200k, 6 langues)"]
    end
    subgraph INGESTION[" Pipeline de traitement"]
        CONNECT[" Import des données"]
        PREPROC[" Préparation et enrichissement"]
        CHUNK[" Découpage en segments"]
    end
    subgraph RECHERCHE[" Moteur de recherche"]
        QREWRITE[" Traitement de la requête"]
        HYBRID[" Recherche combinée (texte + contexte)"]
        RERANK[" Classement des résultats"]
    end
    subgraph GENERATION[" Génération de réponse"]
        LLM[" Synthèse via LLM"]
    end
    subgraph BACKEND[" Serveur web"]
        API[" FastAPI REST"]
    end
    subgraph FRONTEND[" Interface utilisateur"]
        UI[" Streamlit / HTML"]
    end
    SOURCES --> CONNECT --> PREPROC --> CHUNK
    CHUNK --> RECHERCHE --> LLM --> BACKEND --> UI
```

---

## Structure du projet

```
RAG-Time/
├── api.py                     # API REST FastAPI (uvicorn)
├── app.py                     # Frontend Streamlit
├── evaluate_rag.py            # Script d'évaluation (LLM-as-judge)
├── requirements.txt           # Dépendances Streamlit / évaluation
├── requirements_api.txt       # Dépendances API FastAPI
├── .env.example               # Variables d'environnement à configurer
│
├── src/                       # Bibliothèque métier
│   ├── embedder.py            # Adaptateur embeddings (HF API + local)
│   ├── ingest.py              # Pipeline d'ingestion CSV + documents
│   ├── rag.py                 # Moteur RAG (retrieval + LLM)
│   ├── evaluate.py            # Métriques d'évaluation (retrieval + LLM-as-judge)
│   └── stats/
│       └── generate_stats.py  # Analyse exploratoire (EDA)
│
├── scripts/                   # Scripts d'administration (CLI)
│   ├── import_data.py         # Téléchargement du dataset Kaggle
│   ├── reindex.py             # Ré-indexation complète du vectorstore
│   ├── rebuild_vectorstore.py # Migration de modèle d'embedding
│   └── archive/               # Brouillons / expérimentations
│
├── tests/                     # Tests manuels et de régression
│   ├── test_rag.py
│   ├── test_model.py
│   ├── test_multilingual_pipeline.py
│   └── test_openrouter.py
│
├── docs/                      # Documentation du projet
│   ├── cadrage.md             # Document de cadrage
│   ├── architecture_rag_v1.md # Schémas d'architecture (Mermaid)
│   ├── veille_rag.md          # Note de veille technologique
│   └── eda_rapport.md         # Rapport d'analyse exploratoire
│
├── notebooks/                 # Notebooks d'exploration
│   ├── rag_pipeline_report.ipynb
│   └── exploration.ipynb
│
├── data/
│   ├── raw/                   # Données brutes (CSV)
│   ├── processed/             # Statistiques et graphiques EDA
│   ├── vectorstore/           # Index FAISS principal (BGE-M3)
│   └── vectorstore_multilingual_test/  # Index de test (MiniLM)
│
├── evaluation/                # Rapports d'évaluation (JSON, CSV, PNG)
└── static/                    # Frontend HTML (servi par FastAPI)
```

---

## Prérequis

- Python 3.11+
- Clé API [OpenRouter](https://openrouter.ai/) (accès LLM)
- Clé API [HuggingFace](https://huggingface.co/settings/tokens) (accès embeddings serverless)

---

## Installation

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd rag_time

# 2. Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Editer .env et renseigner les clés API
```

---

## Configuration

Copier `.env.example` vers `.env` et renseigner les valeurs :

| Variable | Description | Exemple |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Clé API OpenRouter (LLM) | `sk-or-...` |
| `HF_API_KEY` | Clé API HuggingFace (embeddings) | `hf_...` |
| `EMBED_MODEL` | Modèle d'embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `LLM_MODEL` | Modèle LLM par défaut | `openai/gpt-4o-mini` |
| `TOP_K` | Chunks récupérés par requête | `5` |

---

## Ingestion des données

```bash
# 1. Télécharger le dataset Kaggle (nécessite kagglehub)
python scripts/import_data.py

# 2. Indexer les tickets (première fois ou après changement de données)
python scripts/reindex.py

# 3. Optionnel : limiter à 10 000 tickets pour le développement
python scripts/reindex.py --max-rows 10000
```

---

## Lancement

### Frontend Streamlit

```bash
streamlit run app.py
```

Ouvre [http://localhost:8501](http://localhost:8501)

Fonctionnalités :
- Indexation à la demande depuis la sidebar
- Upload de documents additionnels (PDF, TXT, MD)
- 10 filtres métadonnées (produit, catégorie, priorité, statut, langue, région...)
- Chat avec affichage des sources et scores de similarité

### API REST FastAPI

```bash
uvicorn api:app --reload --port 8000
```

Documentation automatique : [http://localhost:8000/docs](http://localhost:8000/docs)

Endpoints :
- `GET  /api/health` — état de l'API et de l'index
- `GET  /api/info` — infos sur le vectorstore chargé
- `POST /api/chat` — requête RAG (retrieval + génération LLM)

---

## Évaluation

```bash
# Évaluation complète (50 questions générées, LLM-as-judge)
python evaluate_rag.py

# Options avancées
python evaluate_rag.py --n-samples 100 --top-k 5
python evaluate_rag.py --retrieval-only           # sans appel LLM
python evaluate_rag.py --judge-model openai/gpt-4o
```

Rapports générés dans `evaluation/` (JSON + CSV + graphiques PNG).

---

## Tests

```bash
# Test rapide du moteur RAG
python tests/test_rag.py

# Test des modèles OpenRouter disponibles
python tests/test_model.py

# Test du pipeline multilingue (requiert dataset Kaggle)
python tests/test_multilingual_pipeline.py
```

---

## Stratégie d'industrialisation

Pour une mise en production :

1. **Ingestion automatisée** — tâche cron quotidienne sur `scripts/reindex.py` avec filtre de date.
2. **Mise à jour temps réel** — webhook Zendesk/Jira → endpoint FastAPI → ingestion + re-indexation.
3. **Migration de modèle** — `scripts/rebuild_vectorstore.py` pour changer d'embedding sans re-parser le CSV.
4. **Conteneurisation** — Dockerfile (`uvicorn api:app --host 0.0.0.0 --port 8000`).
5. **RBAC (V2)** — JWT + rôles pour cloisonner l'accès par périmètre utilisateur.
6. **Recherche hybride (V2)** — OpenSearch ou Qdrant (BM25 + vectoriel + filtres natifs).

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/cadrage.md](docs/cadrage.md) | Document de cadrage — besoin, périmètre, architecture, risques |
| [docs/architecture_rag_v1.md](docs/architecture_rag_v1.md) | Schémas d'architecture complets (Mermaid) |
| [docs/veille_rag.md](docs/veille_rag.md) | Note de veille technologique (embeddings, RAG, évaluation) |
| [docs/eda_rapport.md](docs/eda_rapport.md) | Analyse exploratoire du dataset |

---

## Choix techniques clés

| Composant | Choix | Raison |
|-----------|-------|--------|
| **Embeddings** | HF Inference API (MiniLM) | Zéro RAM local, multilingue, gratuit |
| **Embeddings (alt.)** | BGE-M3 local | Haute performance, 1024 dims, 8192 tokens |
| **Vector store** | FAISS IndexFlatIP | Recherche exacte, ~1ms, pas de dépendance C++ |
| **LLM** | OpenRouter (gpt-4o-mini) | API unifiée, multi-modèles, coûts maîtrisés |
| **Backend** | FastAPI | Async, léger, doc auto (OpenAPI) |
| **Frontend** | Streamlit | Rapide à développer, filtres intégrés |
