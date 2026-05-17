# Veille technologique complète — Projet RAG-Time

> **Projet M2 IA — RAG-Time**
> Note de veille technologique, étude du besoin métier et analyse du système d'information
> Avril 2026

---

# PARTIE 1 — Note de veille technologique

---

## 1.1 — L'indexation sémantique par plongement (embeddings)

### 1.1.1 — Principe du plongement vectoriel : de word2vec aux modèles modernes

Le **plongement vectoriel** (embedding) consiste à projeter des unités linguistiques (mots, phrases, documents) dans un espace vectoriel continu de dimension fixe, de sorte que la proximité géométrique entre vecteurs reflète la proximité sémantique entre les textes qu'ils représentent.

#### Généalogie des modèles d'embeddings

| Époque | Modèle / Approche | Principe | Limites |
|--------|-------------------|----------|---------|
| 2013 | **word2vec** (Mikolov et al.) | Apprentissage de vecteurs de mots par prédiction de contexte (CBOW, Skip-gram). Chaque mot reçoit un vecteur unique dans ℝ^d (d = 100–300). | Vecteur statique : un seul vecteur par mot, sans prise en compte de la polysémie ni du contexte. |
| 2014 | **GloVe** (Pennington et al.) | Factorisation de la matrice de co-occurrences globale. Combine statistiques globales et prédiction locale. | Mêmes limites que word2vec : pas de contextualisation. |
| 2017 | **Transformer** (Vaswani et al., *Attention Is All You Need*) | Architecture attention multi-têtes, encodage positionnel. Permet la contextualisation de chaque token par rapport à la séquence entière. | Modèle de base, pas directement conçu pour produire des embeddings de phrases. |
| 2018 | **ELMo** (Peters et al.) | Embeddings contextualisés issus de BiLSTM profonds. Chaque occurrence d'un mot produit un vecteur différent selon le contexte. | Architecture LSTM, moins parallélisable que le Transformer. |
| 2018–2019 | **BERT** (Devlin et al.) | Pré-entraînement bidirectionnel Transformer par Masked Language Model + Next Sentence Prediction. Embeddings contextuels de haute qualité. | BERT produit des embeddings de tokens ; l'agrégation naïve (mean pooling du dernier layer) donne des embeddings de phrases de qualité médiocre (Reimers & Gurevych, 2019). |
| 2019 | **Sentence-BERT (SBERT)** (Reimers & Gurevych) | Fine-tuning de BERT/RoBERTa en réseau siamois sur des paires de phrases (NLI, STS). Produit des embeddings de phrases directement comparables par similarité cosinus. | Entraîné principalement en anglais ; performances dégradées en multilingue sans adaptation. |
| 2022–2023 | **E5** (Wang et al., Microsoft) | Modèle entraîné avec des paires (query, passage) à grande échelle via contrastive learning. Versions multilingues disponibles (e5-multilingual-large). | Nécessite un préfixe spécifique (`query:` / `passage:`) pour des performances optimales. |
| 2023 | **BGE** (BAAI) | Beijing Academy of AI. Entraînement contrastif + instruction-tuning. BGE-M3 : modèle multilingue, multi-fonctionnel (dense + sparse + multi-vector). | Modèle volumineux (568M paramètres pour M3). |
| 2024 | **Nomic Embed** (Nomic AI) | Modèle ouvert (code + poids + données d'entraînement publiés), 137M paramètres, 8192 tokens de contexte, performances compétitives sur MTEB. | Moins performant que les plus gros modèles sur certaines tâches de classification. |
| 2024–2025 | **text-embedding-3** (OpenAI), **embed-v3** (Cohere) | Modèles propriétaires accessibles par API. Dimensions ajustables (Matryoshka embeddings pour OpenAI). Support multilingue natif. | Dépendance à un fournisseur, coût récurrent, données transitant par un tiers. |

> **Source** : Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*, arXiv:1301.3781 ; Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers*, arXiv:1810.04805 ; Reimers & Gurevych (2019), *Sentence-BERT*, arXiv:1908.10084 ; Wang et al. (2022), *Text Embeddings by Weakly-Supervised Contrastive Pre-training*, arXiv:2212.03533 ; Chen et al. (2024), *BGE M3-Embedding*, arXiv:2402.03216.

---

### 1.1.2 — Espaces vectoriels et similarité cosinus

Une fois les textes projetés dans un espace vectoriel $\mathbb{R}^d$, la notion de **similarité sémantique** se traduit en **proximité géométrique**. Plusieurs métriques sont utilisables :

**Similarité cosinus.** Mesure l'angle entre deux vecteurs, indépendamment de leur norme :

$$\text{cos\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

- Valeur dans $[-1, 1]$ (dans la pratique, les embeddings normalisés donnent des valeurs dans $[0, 1]$).
- C'est la métrique la plus utilisée pour les embeddings textuels, car elle est invariante à la longueur du vecteur (un texte plus long ne produit pas mécaniquement un vecteur de plus grande norme après normalisation).

**Distance euclidienne (L2).** Mesure la distance absolue dans l'espace :

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}$$

- Sensible à la norme des vecteurs. Si les embeddings sont normalisés L2 (ce qui est recommandé), la distance euclidienne est monotoniquement liée à la similarité cosinus : $d^2 = 2(1 - \text{cos\_sim})$.

**Produit scalaire (dot product).** Utilisé quand la norme porte une information utile (par exemple, saillance ou confiance) :

$$\text{dot}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{d} a_i \cdot b_i$$

> **Recommandation projet** : pour le pipeline RAG-Time, on utilisera la **similarité cosinus** avec des embeddings normalisés L2, ce qui correspond au paramétrage par défaut d'OpenSearch k-NN avec le space type `cosinesimil`.

> **Source** : OpenSearch Documentation, *Vector Search — Space types*, https://docs.opensearch.org/latest/vector-search/

---

### 1.1.3 — Modèles d'embeddings pertinents pour le français et l'anglais

Le choix du modèle d'embedding est critique pour la qualité du retrieval. Pour un contexte B2B franco-anglais (tickets de support rédigés tantôt en français, tantôt en anglais, avec du jargon technique), les critères de sélection sont :

1. **Support multilingue** (français + anglais au minimum)
2. **Performance sur les benchmarks de retrieval** (MTEB, BEIR)
3. **Longueur de contexte** (les tickets peuvent être longs : descriptions + historiques)
4. **Dimensionnalité** (compromis qualité / coût de stockage / vitesse de recherche)
5. **Licence** et possibilité d'auto-hébergement

#### Tableau comparatif des modèles d'embeddings

| Modèle | Éditeur | Params | Dims | Contexte max | Multilingue | Licence | MTEB Retrieval (avg) | Notes |
|--------|---------|--------|------|-------------|-------------|---------|---------------------|-------|
| **e5-multilingual-large** | Microsoft | 560M | 1024 | 512 tokens | ✅ 100+ langues | MIT | ~52 | Solide en multilingue, contexte limité |
| **BGE-M3** | BAAI | 568M | 1024 | 8192 tokens | ✅ 100+ langues | MIT | ~55 | Multi-fonctionnel : dense + sparse + colbert. Contexte long. |
| **nomic-embed-text-v1.5** | Nomic AI | 137M | 768 | 8192 tokens | ✅ (anglais dominant) | Apache 2.0 | ~50 | Léger, reproductible (open data), Matryoshka |
| **multilingual-e5-large-instruct** | Microsoft | 560M | 1024 | 512 tokens | ✅ 100+ langues | MIT | ~54 | Suit les instructions de tâche dans le prompt |
| **text-embedding-3-large** | OpenAI | N/A | 3072 (ajustable) | 8191 tokens | ✅ | Propriétaire | ~55 | Matryoshka (dims réduites possibles), excellent en anglais |
| **embed-v3** | Cohere | N/A | 1024 | 512 tokens | ✅ 100+ langues | Propriétaire | ~55 | Input types dédiés (search_query, search_document) |
| **CamemBERT** / **FlauBERT** | INRIA / CNRS | 110–340M | 768 | 512 tokens | 🇫🇷 uniquement | MIT | ~38 (fr) | Spécialisé français, mais non multilingue |

> **Recommandation projet** : **BGE-M3** est le candidat le plus polyvalent pour RAG-Time — il offre un contexte long (8192 tokens), un support multilingue robuste, et la capacité de produire à la fois des vecteurs denses et des vecteurs sparse (utiles pour la recherche hybride). En alternative API, **text-embedding-3-large** via OpenRouter offre d'excellentes performances avec la flexibilité Matryoshka.

> **Source** : Muennighoff et al. (2023), *MTEB: Massive Text Embedding Benchmark*, arXiv:2210.07316 ; Chen et al. (2024), *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity*, arXiv:2402.03216.

---

### 1.1.4 — Embeddings via API vs. modèles auto-hébergés

| Critère | API (OpenRouter, OpenAI, Cohere) | Auto-hébergé (sentence-transformers, vLLM) |
|---------|----------------------------------|---------------------------------------------|
| **Mise en œuvre** | Immédiate — appel HTTP | Nécessite infrastructure GPU (ou CPU pour les petits modèles) |
| **Coût** | Variable (~$0.02–0.13 / 1M tokens) | Coût fixe infra + maintenance |
| **Latence** | Réseau-dépendante (~50–200 ms) | Très faible en local (~5–20 ms / batch) |
| **Confidentialité** | Données transitent par un tiers | Données restent on-premise |
| **Scalabilité** | Élastique (limites rate) | Dépend du sizing GPU |
| **Mise à jour modèle** | Transparente (risque de breaking change) | Contrôlée (on choisit la version) |
| **Reproductibilité** | Risque de dérive si le fournisseur met à jour le modèle à version constante | Garantie par le versioning local |

**Enjeu RAG-Time** : dans un contexte B2B, la question de la **confidentialité des données** est primordiale. Les tickets de support contiennent potentiellement des informations clients sensibles. Deux stratégies :

1. **MVP** : utiliser l'API OpenRouter pour le text-embedding-3-large (rapidité de développement, pas d'infra GPU à provisionner). S'assurer que les données sont anonymisées (suppression PII) avant envoi.
2. **Production** : migrer vers un modèle auto-hébergé (BGE-M3 via sentence-transformers ou TEI — Text Embeddings Inference de Hugging Face) pour maîtriser totalement le cycle de vie des données.

> **Source** : Hugging Face, *Text Embeddings Inference (TEI)*, https://huggingface.co/docs/text-embeddings-inference

---

### 1.1.5 — Construction d'un index vectoriel : HNSW, FAISS, OpenSearch k-NN

La recherche du plus proche voisin exact (brute-force) dans un espace à haute dimension ($d \geq 768$) a une complexité $O(n \cdot d)$, ce qui est prohibitif pour des bases de centaines de milliers de documents. On utilise donc des algorithmes de recherche approximative (ANN — Approximate Nearest Neighbors).

#### Algorithmes ANN principaux

| Algorithme | Principe | Complexité recherche | Avantages | Inconvénients |
|-----------|---------|---------------------|-----------|---------------|
| **HNSW** (Hierarchical Navigable Small World) | Graphe de voisinage multi-couches. Navigation hiérarchique du graphe le plus grossier au plus fin. | $O(\log n)$ | Très bon recall (>95%), performant en lecture, pas besoin de quantification | Coût mémoire élevé (index entier en RAM), construction lente |
| **IVF** (Inverted File Index) | Quantification de Voronoï : partitionnement de l'espace en cellules, recherche limitée aux cellules proches de la requête. | $O(n_{probe} \cdot n_{cell})$ | Mémoire réduite, bonne scalabilité | Recall dépend du nombre de cellules sondées |
| **IVF-PQ** (IVF + Product Quantization) | Combinaison de IVF avec une compression des vecteurs par quantification produit. | $O(n_{probe} \cdot n_{cell})$ | Très compact en mémoire | Perte de précision due à la quantification |
| **FAISS** (Facebook AI Similarity Search) | Librairie C++ / Python (Meta). Supporte HNSW, IVF, PQ, et leurs combinaisons. | Variable | Référence de l'industrie, très optimisé GPU | Librairie bas niveau, pas de persistance native, pas de filtre métadonnées |
| **ScaNN** (Google) | Quantification anisotrope + partitionnement. | Variable | Très rapide en production Google | Moins universel que FAISS |

#### OpenSearch k-NN

OpenSearch intègre nativement la recherche vectorielle via son plugin **k-NN** :

- **Moteurs supportés** : NMSLIB (HNSW), FAISS (HNSW, IVF), Lucene (HNSW)
- **Méthode recommandée** : HNSW via le moteur Lucene (intégré nativement depuis OpenSearch 2.x, pas de plugin externe requis) ou NMSLIB pour des cas à très haute volumétrie
- **Paramètres clés HNSW** :
  - `m` : nombre de connexions par nœud (défaut : 16). Plus élevé → meilleur recall, plus de mémoire
  - `ef_construction` : taille de la liste de candidats à la construction (défaut : 100)
  - `ef_search` : taille de la liste de candidats à la recherche (défaut : 100). Plus élevé → meilleur recall, plus lent
- **Space types** : `cosinesimil`, `l2`, `innerproduct`
- **Filtrage métadonnées** : OpenSearch permet de combiner la recherche k-NN avec des filtres sur les champs métadonnées (client_id, category, product, date) via des query DSL standards — essentiel pour le cloisonnement des données dans RAG-Time.

**Exemple de mapping OpenSearch pour RAG-Time :**

```json
{
  "mappings": {
    "properties": {
      "content": { "type": "text", "analyzer": "french" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "lucene",
          "parameters": { "m": 16, "ef_construction": 128 }
        }
      },
      "client_id": { "type": "keyword" },
      "category": { "type": "keyword" },
      "product": { "type": "keyword" },
      "created_at": { "type": "date" },
      "ticket_id": { "type": "keyword" },
      "resolution": { "type": "text", "analyzer": "french" }
    }
  }
}
```

> **Source** : OpenSearch Documentation, *k-NN plugin*, https://docs.opensearch.org/latest/vector-search/ ; Malkov & Yashunin (2018), *Efficient and robust approximate nearest neighbor search using HNSW graphs*, arXiv:1603.09320 ; Johnson et al. (2019), *Billion-scale similarity search with GPUs (FAISS)*, arXiv:1702.08734.

---

### 1.1.6 — Hybrid Search : BM25 + vecteurs et score fusion

La recherche hybride combine deux paradigmes complémentaires :

| Paradigme | Forces | Faiblesses |
|-----------|--------|------------|
| **BM25 (full-text)** | Excellent pour les correspondances lexicales exactes (noms de produits, codes d'erreur, identifiants techniques). Rapide, mature, bien compris. | Ne capture pas la sémantique : « l'imprimante ne fonctionne plus » ≠ « problème d'impression » |
| **Recherche vectorielle** | Capture la similarité sémantique, gère la paraphrase et le multilinguisme. | Peut manquer les correspondances exactes (un code d'erreur spécifique). Sensible au bruit dans les embeddings. |

**La combinaison des deux** est systématiquement recommandée dans la littérature RAG (Gao et al., 2024 ; voir aussi les benchmarks de Pinecone et Weaviate qui montrent un gain de 5–15% de recall en hybride vs. vecteur seul).

#### Méthodes de fusion de scores

**1. Reciprocal Rank Fusion (RRF)**

C'est la méthode la plus répandue et la plus robuste :

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

où $R$ est l'ensemble des listes de résultats (BM25, vecteur), $\text{rank}_r(d)$ est le rang du document $d$ dans la liste $r$, et $k$ est une constante (typiquement $k = 60$).

- **Avantage** : ne nécessite pas de normalisation des scores (travaille sur les rangs), robuste aux différences d'échelle entre BM25 et cosinus.
- **Inconvénient** : ne tient pas compte de la marge entre les scores (un document en position 1 avec un score écrasant est traité de la même manière qu'un document en position 1 avec un score marginal).

**2. Normalisation linéaire + pondération**

$$\text{score}(d) = \alpha \cdot \text{norm}(\text{BM25}(d)) + (1 - \alpha) \cdot \text{norm}(\text{cosine}(d))$$

avec $\alpha \in [0, 1]$ à calibrer empiriquement. La normalisation peut être min-max sur le batch de résultats.

- **Avantage** : permet de pondérer l'importance relative du lexical vs. sémantique.
- **Inconvénient** : sensible à la distribution des scores, nécessite calibration.

**3. Implémentation dans OpenSearch**

OpenSearch supporte nativement la recherche hybride via les **search pipelines** et le processeur `normalization-processor` :

```json
{
  "description": "Hybrid search pipeline",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": { "technique": "min_max" },
        "combination": { "technique": "arithmetic_mean", "parameters": { "weights": [0.3, 0.7] } }
      }
    }
  ]
}
```

Ou via RRF directement :

```json
{
  "combination": { "technique": "rrf", "parameters": { "rank_constant": 60 } }
}
```

> **Source** : Cormack et al. (2009), *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR ; OpenSearch Documentation, *Hybrid search*, https://docs.opensearch.org/latest/search-plugins/hybrid-search/ ; Gao et al. (2024), *Retrieval-Augmented Generation for Large Language Models: A Survey*, arXiv:2312.10997.

---

### 1.1.7 — Limites et défis de l'indexation sémantique

| Défi | Description | Mitigation |
|------|-------------|------------|
| **Dimensionnalité élevée** | Les modèles modernes produisent des vecteurs en 768–3072 dimensions. Coût mémoire : 1M vecteurs × 1024 dims × 4 octets = ~4 Go. | Utiliser la quantification (int8, binary), les embeddings Matryoshka (réduire à 256–512 dims avec perte marginale), ou la Product Quantization. |
| **Malédiction de la dimensionnalité** | En haute dimension, les distances entre points tendent à se concentrer, réduisant le pouvoir discriminant des métriques. | Normalisation L2, choix de dimensionnalité adaptée au corpus. |
| **Out-of-distribution (OOD)** | Un modèle entraîné sur des textes généralistes peut mal représenter le jargon métier spécifique (codes produits, acronymes internes). | Fine-tuning du modèle d'embedding sur des paires (question, passage) issues du domaine. Alternatively, enrichir les chunks avec des métadonnées explicites. |
| **Drift temporel** | Les nouveaux tickets utilisent un vocabulaire qui évolue (nouveaux produits, nouvelles catégories). Les embeddings calculés à un instant $t$ peuvent devenir moins pertinents à $t+1$. | Re-indexation périodique (ex. hebdomadaire pour les anciens chunks), indexation incrémentale en quasi-temps réel pour les nouveaux documents. |
| **Coût de re-indexation** | Recalculer les embeddings de toute la base est coûteux (temps + compute + coût API). Pour 200k tickets × 3 chunks × $0.02/1M tokens ≈ quelques dollars, mais pour des bases plus volumineuses, cela devient significatif. | Indexation incrémentale, mise en cache des embeddings, utilisation de modèles légers pour le re-indexing courant. |
| **Multilinguisme asymétrique** | Même les modèles multilingues performent mieux en anglais qu'en français sur les benchmarks retrieval. Écart de 5–10 points sur MTEB. | Évaluer spécifiquement les performances sur un échantillon de tickets français. Envisager un fine-tuning sur des données françaises. |
| **Tokens spéciaux et jargon** | Les tokenizers BPE découpent les termes techniques inconnus en sous-mots, diluant l'information sémantique. | Compléter la recherche vectorielle par une recherche BM25 (hybrid search) qui capture les correspondances exactes. |

---

## 1.2 — Les différentes sortes de RAG

Le paradigme RAG (Retrieval-Augmented Generation), introduit par Lewis et al. (2020), fournit des connaissances externes à un LLM au moment de l'inférence, plutôt que de les stocker entièrement dans ses paramètres. Depuis cette publication fondatrice, le concept a considérablement évolué. La taxonomie suivante s'appuie sur les surveys de Gao et al. (2024, arXiv:2312.10997) et Fan et al. (2025, arXiv:2506.00054).

---

### 1.2.1 — RAG Naïf (Naive RAG)

#### Principe

Pipeline linéaire en trois étapes :

```
Question utilisateur → Retrieval (top-k documents) → Génération (LLM avec contexte)
```

1. **Indexation** : les documents sont segmentés en chunks, vectorisés, puis stockés dans un index vectoriel.
2. **Retrieval** : la question de l'utilisateur est vectorisée avec le même modèle, puis les k plus proches voisins sont récupérés.
3. **Génération** : les k chunks sont insérés dans le prompt du LLM avec la question, et le modèle produit une réponse.

#### Cas d'usage

- Prototypage rapide, proof-of-concept
- Corpus homogènes de petite à moyenne taille
- Questions simples, factuelles

#### Avantages

- Simplicité d'implémentation (quelques lignes avec LangChain ou LlamaIndex)
- Faible coût de développement et d'infra
- Facile à comprendre et à débugger

#### Limites

- **Qualité de retrieval limitée** : correspondance query-document uniquement par similarité vectorielle, pas de réécriture ni de raffinement.
- **Lost in the middle** : les LLMs tendent à sur-pondérer le début et la fin du contexte, négligeant les chunks intermédiaires (Liu et al., 2023, *Lost in the Middle*, arXiv:2307.03172).
- **Pas de contrôle qualité** : aucun mécanisme pour détecter si les chunks récupérés sont réellement pertinents.
- **Hallucinations** : le LLM peut ignorer les chunks fournis et répondre à partir de ses connaissances paramétriques, surtout si le contexte est bruité.
- **Pas de gestion de la multi-step reasoning** : une seule passe de retrieval.

> **Source** : Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, arXiv:2005.11401 ; Gao et al. (2024), arXiv:2312.10997.

---

### 1.2.2 — RAG Avancé (Advanced RAG)

Le RAG avancé améliore chaque étape du pipeline naïf par des techniques d'optimisation en pré-retrieval, retrieval et post-retrieval.

#### A. Optimisations pré-retrieval

**Query Rewriting.** La requête brute de l'utilisateur est souvent ambiguë, incomplète ou mal formulée. Le query rewriting utilise un LLM pour reformuler la question en une version plus précise et mieux adaptée au retrieval.

Exemple :
- Question brute : *« l'imprimante marche plus »*
- Réécriture : *« L'imprimante ne fonctionne plus. Quels sont les tickets de support relatifs à un dysfonctionnement d'imprimante ou un problème d'impression ? »*

**HyDE (Hypothetical Document Embeddings)** (Gao et al., 2022). Au lieu de chercher avec l'embedding de la question, on demande au LLM de générer un *document hypothétique* qui répondrait à la question, puis on cherche les voisins de ce document hypothétique. L'intuition est que l'embedding d'une réponse sera plus proche des documents pertinents que l'embedding de la question.

$$\text{query} \xrightarrow{\text{LLM}} \text{hypothetical\_doc} \xrightarrow{\text{embed}} \text{vector} \xrightarrow{\text{k-NN}} \text{results}$$

- **Avantage** : comble le *query-document gap* (asymétrie sémantique entre questions et réponses).
- **Limite** : ajoute un appel LLM (latence + coût), le document hypothétique peut être faux et orienter vers des résultats hors sujet.

**Query Expansion / Decomposition.** Décomposer une question complexe en sous-questions, chacune faisant l'objet d'un retrieval séparé, puis fusionner les résultats.

> **Source** : Gao et al. (2022), *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*, arXiv:2212.10496.

#### B. Retrieval amélioré

**Re-ranking (Cross-Encoders).** Après le retrieval initial (qui doit être rapide — top-100 ou top-50), un cross-encoder réévalue la pertinence de chaque (question, chunk) paire. Contrairement aux bi-encoders qui produisent des embeddings indépendants puis comparent, les cross-encoders prennent en entrée la concaténation (question + chunk) et produisent directement un score de pertinence.

| Modèle de reranking | Éditeur | Contexte max | Performance |
|---------------------|---------|-------------|-------------|
| **ms-marco-MiniLM-L-12-v2** | cross-encoder (HF) | 512 tokens | Référence, rapide |
| **bge-reranker-v2-m3** | BAAI | 8192 tokens | Multilingue, contexte long |
| **Cohere Rerank v3** | Cohere | 4096 tokens | API, très performant |
| **Jina Reranker v2** | Jina AI | 8192 tokens | Multilingue, open source |

**Hybrid Search.** Comme détaillé en §1.1.6, la combinaison BM25 + vecteurs avec RRF ou pondération linéaire.

**Metadata Filtering.** Filtrer les résultats par métadonnées structurées (client_id, category, product, date range) *avant* ou *pendant* la recherche k-NN. OpenSearch le permet nativement via les query filters combinés aux requêtes k-NN.

#### C. Optimisations post-retrieval

**Context Compression.** Les k chunks retournés contiennent souvent des passages non pertinents. Un modèle de compression (LLMChainExtractor dans LangChain, ou un appel LLM dédié) extrait uniquement les passages pertinents de chaque chunk.

**Lost-in-the-Middle Mitigation.** Réordonner les chunks de sorte que les plus pertinents soient au début et à la fin du contexte (pas au milieu), conformément aux findings de Liu et al. (2023).

**Map-Reduce / Refine.** Pour les questions nécessitant de synthétiser de nombreux documents :
- **Map** : le LLM traite chaque chunk séparément et produit une réponse partielle.
- **Reduce** : un second appel LLM combine toutes les réponses partielles en une synthèse finale.

**Avantages du RAG Avancé par rapport au Naïf** : amélioration significative du recall (+10–25% selon les benchmarks), réduction des hallucinations, meilleure gestion des questions complexes.

**Limites** : complexité accrue du pipeline, latence additionnelle (chaque optimisation ajoute un appel modèle ou une étape de traitement), calibration des hyperparamètres (poids hybrid search, nombre de chunks avant/après reranking).

> **Source** : Gao et al. (2024), arXiv:2312.10997 ; Liu et al. (2023), *Lost in the Middle*, arXiv:2307.03172 ; Ma et al. (2023), *Query Rewriting for Retrieval-Augmented Large Language Models*, arXiv:2305.14283.

---

### 1.2.3 — RAG Modulaire (Modular RAG)

#### Principe

Le RAG modulaire conçoit le pipeline non pas comme une séquence linéaire rigide, mais comme un assemblage de **modules interchangeables** orchestrés dynamiquement :

```
[Search Module] ⇄ [Memory Module] ⇄ [Routing Module] ⇄ [Predict Module] ⇄ [Task Adapter]
```

Chaque module peut être remplacé, ajouté ou supprimé indépendamment. Un **routeur** (router) décide, pour chaque requête, quel chemin emprunter dans le pipeline.

#### Modules types

| Module | Rôle | Exemples d'implémentation |
|--------|------|--------------------------|
| **Search** | Récupération de contexte depuis différentes sources | BM25, k-NN, recherche SQL, API externe |
| **Memory** | Stockage de l'historique conversationnel ou des résultats intermédiaires | Redis, buffer mémoire |
| **Routing** | Aiguillage de la requête vers le pipeline adapté selon le type de question | Classification LLM, règles métier |
| **Predict** | Génération de la réponse | LLM (Claude, GPT-4, Mistral) |
| **Fusion** | Fusion des résultats de plusieurs modules Search | RRF, re-ranking |
| **Validate** | Vérification de la cohérence / fidélité de la réponse | LLM critique, règles heuristiques |

#### Cas d'usage

- Systèmes RAG d'entreprise multi-sources (tickets + ERP + CRM)
- Cas où différents types de questions nécessitent différents pipelines (ex. : question factuelle → retrieval simple ; question analytique → SQL + agrégation)

#### Avantages

- Flexibilité maximale, évolutif
- Permet l'optimisation indépendante de chaque composant
- Facilite les tests A/B sur des modules spécifiques

#### Limites

- Complexité d'ingénierie significative
- Nécessite un framework orchestrateur (LangGraph, DSPy, Haystack)
- Le routeur lui-même peut être un point de défaillance

> **Source** : Gao et al. (2024), arXiv:2312.10997 (section 4 — Modular RAG) ; Khattab et al. (2022), *DSPy: Compiling Declarative Language Model Calls*, arXiv:2310.03714.

---

### 1.2.4 — Self-RAG

#### Principe

Proposé par Asai et al. (2023), **Self-RAG** entraîne le LLM à émettre des **tokens spéciaux de réflexion** (reflection tokens) qui contrôlent le processus de retrieval et de génération :

1. **`[Retrieve]`** : le modèle décide s'il a besoin de récupérer des informations externes (vs. répondre directement à partir de ses connaissances).
2. **`[IsRel]`** : le modèle évalue si les passages récupérés sont pertinents pour la question.
3. **`[IsSup]`** : le modèle vérifie si sa réponse est fidèle (supportée) par les passages.
4. **`[IsUse]`** : le modèle évalue si la réponse est utile pour l'utilisateur.

Le modèle est fine-tuné avec un training supervisé incluant ces tokens de réflexion, annotés par un *critic model*.

#### Cas d'usage

- Domaines où l'hallucination est critique (médical, juridique, support technique sensible)
- Quand on veut minimiser les appels retrieval inutiles (optimisation de coût)

#### Avantages

- Le modèle ne récupère que quand c'est nécessaire → réduction de latence sur les questions simples
- Auto-évaluation intégrée de la fidélité et de la pertinence
- Performances supérieures aux pipelines RAG naïfs sur plusieurs benchmarks

#### Limites

- Nécessite un **fine-tuning** du LLM (pas un simple prompt engineering), ce qui le rend complexe à déployer avec des modèles propriétaires
- Le critic model utilisé pour l'annotation peut lui-même introduire des biais
- Moins flexible que le RAG modulaire : la logique est câblée dans le modèle

> **Source** : Asai et al. (2023), *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, arXiv:2310.11511.

---

### 1.2.5 — Corrective RAG (CRAG)

#### Principe

Proposé par Yan et al. (2024), **CRAG** ajoute un mécanisme d'auto-correction au pipeline RAG :

1. **Retrieval classique** : récupération des top-k documents.
2. **Évaluation de confiance** : un *retrieval evaluator* (modèle léger) attribue un score de confiance à chaque document récupéré.
3. **Trois cas** :
   - **Confiance haute** → utiliser les documents récupérés tel quel.
   - **Confiance basse** → les documents sont non pertinents. Déclencher une **recherche web** (ou une source alternative) pour compenser.
   - **Confiance ambiguë** → combiner les documents récupérés et les résultats de la recherche de secours.
4. **Knowledge Refinement** : un module décompose les documents en « knowledge strips » et filtre les informations non pertinentes avant de les envoyer au LLM.

#### Cas d'usage

- Corpus incomplets ou hétérogènes en qualité
- Quand la base de connaissances ne couvre pas tous les sujets possibles (ex : nouveau produit non encore documenté)

#### Avantages

- Robustesse face aux échecs de retrieval
- Mécanisme de fallback explicite (recherche web, source alternative)
- Le knowledge refinement améliore la qualité du contexte transmis au LLM

#### Limites

- Ajout de complexité (evaluator + search de secours)
- Le recours au web peut poser des problèmes de confidentialité en entreprise
- Le retrieval evaluator nécessite des données d'entraînement annotées

> **Source** : Yan et al. (2024), *Corrective Retrieval Augmented Generation (CRAG)*, arXiv:2401.15884.

---

### 1.2.6 — Graph RAG

#### Principe

**Graph RAG** enrichit le pipeline RAG avec un **graphe de connaissances** (Knowledge Graph — KG) structurant explicitement les relations entre entités :

1. **Extraction d'entités et de relations** : un LLM ou un modèle NER extrait les entités (produits, clients, erreurs, composants) et leurs relations (« produit X a le problème Y », « résolution Z corrige le problème Y ») depuis les documents.
2. **Construction du graphe** : les entités sont des nœuds, les relations sont des arêtes typées, stockés dans une base de graphe (Neo4j, Amazon Neptune, ou un index de graphe dans OpenSearch).
3. **Retrieval augmenté** : la requête de l'utilisateur est utilisée pour traverser le graphe (graph traversal, subgraph extraction) en plus de la recherche vectorielle classique. Le contexte transmis au LLM inclut à la fois les chunks textuels et les sous-graphes pertinents.

**Variante : community detection.** Microsoft Research (Edge et al., 2024) propose une approche où le graphe de connaissances est partitionné en communautés (clusters de nœuds fortement connectés), et des résumés sont pré-générés pour chaque communauté. Le retrieval porte alors sur ces résumés de communautés, ce qui est efficace pour les questions de synthèse globale (*« Quels sont les principaux problèmes rencontrés par les clients du produit X ? »*).

#### Cas d'usage

- Questions nécessitant un raisonnement multi-hop (« Quel client a eu un problème sur le produit X associé au composant Y ? »)
- Besoins de traçabilité et d'explicabilité (le graphe rend les relations explicites)
- Synthèse globale sur un corpus volumineux

#### Avantages

- Capture les relations structurelles que les embeddings purs ne représentent pas
- Permet des requêtes de type naviguation / exploration (« donne-moi tous les tickets liés à ce client via ce produit »)
- Améliore le multi-hop reasoning

#### Limites

- **Coût d'extraction** : l'extraction d'entités et de relations par LLM est coûteuse et sujette aux erreurs
- **Maintenance du graphe** : le KG doit être mis à jour à chaque nouveau document, avec gestion des conflits et des doublons
- **Complexité d'infrastructure** : ajout d'une base de graphe (Neo4j) au stack, compétences spécifiques requises
- **Scalabilité** : les traversées de graphe peuvent être lentes sur des graphes volumineux

> **Source** : Edge et al. (2024), *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv:2404.16130 ; Pan et al. (2024), *Unifying Large Language Models and Knowledge Graphs: A Roadmap*, arXiv:2306.08302.

---

### 1.2.7 — Agentic RAG

#### Principe

**Agentic RAG** place un **agent autonome** (souvent un LLM doté de capacités de planification et d'utilisation d'outils — *tool use*) au centre du pipeline RAG. L'agent :

1. **Analyse la requête** et élabore un plan de résolution (multi-step reasoning).
2. **Décide quels outils utiliser** : recherche vectorielle, recherche SQL, appel API CRM, calcul, etc.
3. **Itère** : évalue les résultats intermédiaires, décide s'il a assez d'informations, et raffinement sa recherche ou sa réponse.
4. **Produit la réponse finale** en synthétisant les résultats de toutes les étapes.

Frameworks typiques : **LangGraph** (état + transitions), **CrewAI** (multi-agents), **AutoGen** (Microsoft), **OpenAI Assistants**.

#### Cas d'usage

- Questions complexes nécessitant l'accès à plusieurs sources (tickets + CRM + ERP)
- Workflows de résolution guidés (« Cherche les tickets similaires, vérifie le statut du contrat dans le CRM, puis propose une résolution »)
- Automatisation de processus support (triage, escalade, réponse automatique)

#### Avantages

- Très flexible, peut intégrer n'importe quelle source ou outil
- Gère naturellement le raisonnement multi-étapes
- Peut s'auto-corriger et itérer

#### Limites

- **Fiabilité** : les agents LLM peuvent boucler, prendre de mauvaises décisions, ou utiliser des outils de façon incorrecte
- **Coût** : chaque étape d'agencement = un appel LLM supplémentaire
- **Latence** : le raisonnement multi-step peut prendre plusieurs secondes voire dizaines de secondes
- **Observabilité** : difficile à debugger sans un tracing exhaustif (LangSmith, Phoenix)
- **Sécurité** : l'agent a accès à des outils → risque d'actions non souhaitées sans garde-fous stricts

> **Source** : Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models*, arXiv:2210.03629 ; Schick et al. (2023), *Toolformer: Language Models Can Teach Themselves to Use Tools*, arXiv:2302.04761.

---

### 1.2.8 — RAG long-context : le RAG est-il encore nécessaire ?

#### Contexte

Les LLMs récents offrent des fenêtres de contexte considérables :

| Modèle | Fenêtre de contexte | Date |
|--------|---------------------|------|
| GPT-4 Turbo | 128k tokens | Nov 2023 |
| Claude 3.5 Sonnet | 200k tokens | Jun 2024 |
| Gemini 1.5 Pro | 1M–2M tokens | Fév 2024 |
| Claude 4 Opus | 200k tokens | 2025 |
| Llama 3.3 | 128k tokens | Déc 2024 |

Avec 200k tokens, on peut injecter directement ~300 pages de texte dans le prompt. La question se pose alors : **pourquoi s'embarrasser d'un pipeline RAG complexe quand on peut simplement tout mettre dans le contexte ?**

#### Analyse : RAG vs. Long Context

| Critère | Tout dans le contexte | RAG |
|---------|----------------------|-----|
| **Simplicité** | ✅ Pas de pipeline, pas d'index | ❌ Pipeline complet à maintenir |
| **Coût par requête** | ❌ Très élevé (200k tokens × prix/token) | ✅ Faible (seuls les top-k chunks sont envoyés) |
| **Latence** | ❌ Élevée (TTFT proportionnel au contexte) | ✅ Faible (contexte réduit) |
| **Scalabilité** | ❌ Impossible au-delà de la fenêtre (200k ≈ 300 pages, mais une base de 200k tickets fait plusieurs GB) | ✅ Scalable (index vectoriel gère des millions de documents) |
| **Précision** | ⚠️ Dégradation « lost in the middle » accentuée sur les très longs contextes | ✅ Le reranking et le filtrage garantissent la pertinence |
| **Mise à jour** | ✅ Directe (on régénère le prompt avec les données à jour) | ⚠️ Nécessite réindexation |
| **Confidentialité** | ❌ Toutes les données transitent (y compris les non pertinentes) | ✅ Seuls les chunks pertinents sont envoyés |

#### Conclusion

Le long context **ne rend pas le RAG obsolète** — il le **complète** :

- Pour des **corpus volumineux** (>100k documents), le RAG est indispensable pour le filtrage et la scalabilité.
- Pour des **questions ciblées sur un sous-ensemble**, le RAG est plus efficient (coût, latence, précision).
- Le long context est pertinent pour des **analyses globales** sur un petit corpus, ou comme **étape de synthèse** après retrieval (récupérer 50 chunks au lieu de 5, et laisser le LLM long-context trier).
- L'approche hybride « **RAG + long context** » est la plus prometteuse : le RAG filtre un contexte pertinent plus large que dans un pipeline classique (top-50 au lieu de top-5), et le LLM long-context le digère efficacement.

> **Source** : Xu et al. (2024), *Retrieval Head Mechanistically Explains Long-Context Factuality*, arXiv:2404.15574 ; Lee et al. (2024), *Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?*, arXiv:2406.13121.

---

### 1.2.9 — Tableau synthétique des architectures RAG

| Architecture | Complexité | Latence | Qualité retrieval | Qualité génération | Cas d'usage idéal |
|-------------|-----------|---------|-------------------|-------------------|-------------------|
| **Naive RAG** | ⭐ | ⭐ | ⭐⭐ | ⭐⭐ | MVP, prototypage |
| **Advanced RAG** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Production, support technique |
| **Modular RAG** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Entreprise multi-sources |
| **Self-RAG** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Domaines critiques |
| **CRAG** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Corpus incomplets |
| **Graph RAG** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Raisonnement multi-hop |
| **Agentic RAG** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Workflows complexes, multi-outils |
| **Long-context** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Petits corpus, analyse globale |

*(nombre d'étoiles = niveau relatif, plus = plus élevé)*

> **Recommandation RAG-Time** : démarrer en **Advanced RAG** (hybrid search + reranking) pour le MVP tickets, puis évoluer vers un **Modular RAG** pour l'intégration multi-sources (ERP, CRM, PIM, GED), avec un **routeur** qui aiguille les requêtes vers le bon pipeline selon le type de question.

---

## 1.3 — Techniques d'évaluation des solutions RAG et métriques associées

### 1.3.1 — Framework d'évaluation global

L'évaluation d'un système RAG est intrinsèquement **bi-dimensionnelle** : il faut évaluer à la fois la qualité du **retrieval** (les bons documents sont-ils récupérés ?) et la qualité de la **génération** (la réponse est-elle fidèle, pertinente et correcte ?).

Deux approches complémentaires :

| Approche | Description | Avantages | Inconvénients |
|----------|-------------|-----------|---------------|
| **Composant par composant** | Évaluer le retriever et le générateur séparément, avec des métriques dédiées | Permet d'identifier précisément le maillon faible du pipeline | Ne capture pas les interactions entre composants |
| **End-to-end** | Évaluer la réponse finale par rapport à la question et un ground truth | Reflète l'expérience utilisateur réelle | Difficile d'attribuer un échec au retriever ou au générateur |

**Recommandation** : utiliser les deux approches. Évaluer d'abord le retriever isolément (métriques IR), puis le pipeline complet (métriques de génération).

---

### 1.3.2 — Métriques de retrieval

#### Precision@k

Proportion de documents pertinents parmi les k documents récupérés :

$$\text{Precision@k} = \frac{|\text{docs pertinents dans top-k}|}{k}$$

Utile pour évaluer la précision quand on sait que l'utilisateur ne regarde que les k premiers résultats.

#### Recall@k

Proportion de documents pertinents dans le corpus qui ont été récupérés dans le top-k :

$$\text{Recall@k} = \frac{|\text{docs pertinents dans top-k}|}{|\text{total docs pertinents}|}$$

Fondamental pour le RAG : si le retriever rate un document pertinent, le LLM ne pourra jamais le citer.

#### Mean Reciprocal Rank (MRR)

Moyenne, sur l'ensemble des requêtes, de l'inverse du rang du premier document pertinent :

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

Pertinent quand on attend *un seul* document pertinent (ex. : trouver LE ticket qui résout le problème).

#### Normalized Discounted Cumulative Gain (NDCG)

Mesure la qualité du classement en tenant compte de la position des documents pertinents et de la gradation de pertinence :

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

où IDCG est le DCG du classement idéal. Le NDCG pénalise les documents pertinents classés bas.

#### Hit Rate

Proportion de requêtes pour lesquelles au moins un document pertinent est dans le top-k :

$$\text{Hit Rate@k} = \frac{|\{q : \text{au moins 1 doc pertinent dans top-k}\}|}{|Q|}$$

Métrique simple et intuitive, utile pour le suivi en production.

#### Context Relevance

Mesure non standard (introduite par RAGAs et TruLens) : un LLM juge si les chunks récupérés sont pertinents pour répondre à la question. Calculée comme le ratio de phrases / passages du contexte qui sont pertinents pour la question.

> **Source** : Manning et al. (2008), *Introduction to Information Retrieval*, Cambridge University Press ; Es et al. (2023), *RAGAs: Automated Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217.

---

### 1.3.3 — Métriques de génération

#### Faithfulness (Fidélité)

**Définition** : la réponse est-elle fidèle au contexte fourni ? Le modèle ne doit pas inventer d'informations absentes des chunks récupérés.

**Calcul (RAGAs)** :
1. Décomposer la réponse en *claims* (affirmations atomiques).
2. Pour chaque claim, vérifier s'il est supporté par le contexte.
3. Faithfulness = nombre de claims supportés / nombre total de claims.

$$\text{Faithfulness} = \frac{|\text{claims supportés par le contexte}|}{|\text{total claims}|}$$

**Enjeu RAG-Time** : pour un système de support technique, une hallucination (proposer une résolution incorrecte) peut avoir des conséquences opérationnelles graves.

#### Answer Relevance (Pertinence de la réponse)

**Définition** : la réponse répond-elle à la question posée ? Une réponse peut être fidèle au contexte mais hors sujet.

**Calcul (RAGAs)** : un LLM génère n questions hypothétiques à partir de la réponse. La similarité cosinus moyenne entre les embeddings de ces questions générées et la question originale donne le score.

#### Answer Correctness (Exactitude de la réponse)

**Définition** : la réponse est-elle factuellement correcte par rapport à un ground truth ? Combine la fidélité sémantique (F1 sémantique entre la réponse et la vérité terrain) et la fidélité factuelle.

> **Source** : Es et al. (2023), *RAGAs*, arXiv:2309.15217 ; Saad-Falcon et al. (2024), *ARES: An Automated Evaluation Framework for RAG*, arXiv:2311.09476.

---

### 1.3.4 — RAG Triad (TruLens)

TruLens (Truera) propose un cadre d'évaluation articulé autour de trois dimensions formant un triangle vertueux :

```
         Context Relevance
              ╱   ╲
             ╱     ╲
    Groundedness ─── Answer Relevance
```

| Dimension | Question clé | Risque si faible |
|-----------|-------------|------------------|
| **Context Relevance** | Les chunks récupérés sont-ils pertinents pour la question ? | Le LLM reçoit du bruit → hallucination ou réponse hors sujet |
| **Groundedness** (≈ Faithfulness) | La réponse est-elle fondée sur le contexte fourni ? | Le LLM hallucine au-delà du contexte |
| **Answer Relevance** | La réponse répond-elle à la question ? | Réponse correcte mais hors sujet |

**Diagnostic par la triade** :
- Groundedness faible + Context Relevance élevé → problème de génération (le LLM ignore le bon contexte)
- Groundedness faible + Context Relevance faible → problème de retrieval (mauvais chunks → mauvaise réponse)
- Answer Relevance faible → problème de compréhension de la question (rewriting nécessaire)

> **Source** : Mistral AI (2024), *LLM as RAG Judge*, https://mistral.ai/news/llm-as-rag-judge ; TruLens Documentation, https://www.trulens.org/

---

### 1.3.5 — LLM-as-a-Judge

#### Principe

Utiliser un **LLM évaluateur** (souvent un modèle puissant comme GPT-4 ou Claude) pour noter automatiquement les réponses d'un système RAG selon des critères prédéfinis (fidélité, pertinence, complétude).

#### Protocole typique

1. Préparer un jeu de test : {question, contexte récupéré, réponse générée, [ground truth optionnel]}.
2. Construire un prompt d'évaluation structuré avec une grille de notation (ex. : 1–5) et des critères explicites.
3. Soumettre chaque triplet au LLM juge.
4. Agréger les scores.

#### Biais connus

| Biais | Description | Mitigation |
|-------|-------------|------------|
| **Position bias** | Le LLM juge tend à préférer la première option présentée | Randomiser l'ordre des réponses comparées |
| **Verbosity bias** | Préférence pour les réponses plus longues, même si moins précises | Inclure dans le prompt : « la concision est une qualité » |
| **Self-enhancement bias** | Un LLM tend à mieux noter ses propres réponses | Utiliser un modèle juge différent du modèle générateur |
| **Knowledge bias** | Le juge utilise ses propres connaissances plutôt que de se baser sur le contexte fourni | Instruire explicitement : « évalue uniquement par rapport au contexte fourni » |
| **Format bias** | Préférence pour les réponses bien formatées (markdown, listes) indépendamment du contenu | Standardiser le format ou évaluer sur le contenu brut |

#### Calibration

Comparer les scores LLM-as-Judge avec des annotations humaines sur un échantillon (100–500 paires) pour estimer l'accord inter-annotateur (Cohen's κ ou corrélation de Spearman). Un κ > 0.7 est généralement considéré comme acceptable.

> **Source** : Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685 ; Mistral AI (2024), *LLM as RAG Judge*, https://mistral.ai/news/llm-as-rag-judge.

---

### 1.3.6 — Frameworks open source d'évaluation

| Framework | Éditeur | Métriques proposées | Particularités | Licence |
|-----------|---------|-------------------|----------------|---------|
| **RAGAs** | Explodinggradients | Faithfulness, Answer Relevance, Context Precision, Context Recall, Answer Correctness | Le plus utilisé, intégré à LangChain et LlamaIndex | Apache 2.0 |
| **TruLens** | Truera (Snowflake) | RAG Triad (Context Relevance, Groundedness, Answer Relevance) + métriques custom | Dashboard de suivi, traçage des feedbacks | MIT |
| **DeepEval** | Confident AI | 14+ métriques (Faithfulness, Hallucination, Answer Relevance, Contextual Precision/Recall, G-Eval, Summarization, Bias, Toxicity) | Compatible pytest, CI/CD natif, tests unitaires pour LLM | Apache 2.0 |
| **ARES** | Stanford | Prediction-Powered Inference pour estimer les performances avec peu d'annotations humaines | Approche statistiquement rigoureuse (intervalles de confiance) | MIT |
| **Phoenix** (Arize) | Arize AI | Retrieval et generation evals, tracing intégré | Lien fort avec l'observabilité (tracing OpenTelemetry) | BUSL → Apache 2.0 |

> **Source** : Es et al. (2023), *RAGAs*, arXiv:2309.15217 ; Saad-Falcon et al. (2024), *ARES*, arXiv:2311.09476.

---

### 1.3.7 — Benchmarks de référence

| Benchmark | Domaine | Taille | Utilisation RAG |
|-----------|---------|--------|-----------------|
| **BEIR** (Thakur et al., 2021) | 18 datasets IR hétérogènes (NQ, FiQA, SciFact, etc.) | Variable | Évaluation zero-shot des retrievers. Standard de facto pour comparer les modèles d'embedding. |
| **MS MARCO** | Questions Bing + passages web | ~8.8M passages, 1M queries | Entraînement et évaluation de retrievers et re-rankers |
| **MTEB** (Muennighoff et al., 2023) | Benchmark massif multi-tâches pour embeddings | 56+ datasets, 8 tâches | Classement global des modèles d'embedding (leaderboard HuggingFace) |
| **Open RAG Bench** (Vectara) | Évaluation end-to-end de pipelines RAG | Multiples datasets | Focus sur la fidélité (hallucination rate) |
| **KILT** (Petroni et al., 2021) | Knowledge-Intensive Language Tasks | 5 tâches (QA, fact-checking, slot filling) | Évaluation retrieval + knowledge grounding |

> **Source** : Thakur et al. (2021), *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, arXiv:2104.08663 ; Muennighoff et al. (2023), *MTEB*, arXiv:2210.07316.

---

### 1.3.8 — Données synthétiques pour l'évaluation

En l'absence de ground truth complète (cas fréquent en entreprise), on peut **générer automatiquement des paires (question, réponse attendue, chunks pertinents)** à partir du corpus existant :

#### Pipeline de génération synthétique

```
Corpus de chunks → Sélection aléatoire → LLM génère question + réponse → Validation humaine (échantillon)
```

1. **Sélection** : choisir aléatoirement n chunks du corpus (stratification par catégorie recommandée).
2. **Génération** : pour chaque chunk, demander à un LLM de générer :
   - Une question à laquelle le chunk répond
   - La réponse attendue
   - (Optionnel) Des questions de difficulté variable (factuelle, synthétique, multi-hop)
3. **Validation** : un expert humain valide un échantillon (10–20%) pour s'assurer de la qualité.
4. **Utilisation** : ce jeu de test synthétique sert de ground truth pour calculer les métriques de retrieval et de génération.

**Outils** :
- **RAGAs** intègre un module `TestsetGenerator` qui génère automatiquement des questions de complexité variable (simple, reasoning, multi-context).
- **Synthétiseur LlamaIndex** : `RagDatasetGenerator`.
- **Manual + LLM** : prompt spécifique pour un LLM fort (Claude, GPT-4) avec les chunks en contexte.

#### Risques

- **Biais circulaire** : si le même LLM génère les Q&A et les évalue, il y a un risque de complaisance.
- **Couverture inégale** : la génération aléatoire peut sur-représenter certains sujets.
- **Qualité** : les questions synthétiques peuvent être trop faciles ou trop éloignées des vrais patterns de requêtes utilisateurs.

**Mitigation** : croiser les données synthétiques avec des logs de requêtes réelles (quand disponibles) et des annotations humaines sur un échantillon.

---

### 1.3.9 — Cas pratique : évaluer le système RAG-Time (tickets de support) sans ground truth complète

#### Stratégie d'évaluation recommandée pour le projet

**Phase 1 — Évaluation du Retriever (offline, avant intégration LLM)**

1. Générer un jeu de test synthétique de 200–500 paires (question, tickets pertinents) à partir des données :
   - Sélectionner 200 tickets diversifiés (stratifiés par catégorie, produit, sévérité).
   - Pour chaque ticket, générer 1–3 questions variées avec un LLM.
   - Les tickets source constituent le ground truth retrieval.
2. Exécuter le pipeline de retrieval (hybrid search + reranking) sur ces questions.
3. Calculer les métriques : **Recall@5, Recall@10, MRR, NDCG@10, Hit Rate@5**.
4. Objectif : Recall@10 > 85%, MRR > 0.6.

**Phase 2 — Évaluation end-to-end (avec génération LLM)**

1. Sur les 200 paires précédentes, exécuter le pipeline complet (retrieval + génération).
2. Calculer les métriques RAGAs : **Faithfulness, Answer Relevance, Context Precision**.
3. Configurer un LLM-as-Judge (Claude ou GPT-4, différent du modèle de génération) pour scorer :
   - Fidélité (1–5)
   - Pertinence (1–5)
   - Complétude (1–5)
4. Valider le LLM-as-Judge sur 50 paires annotées par un humain → mesurer l'accord (Spearman ρ > 0.7).

**Phase 3 — Évaluation continue (production)**

1. Collecter les feedbacks utilisateurs implicites (clics, taux de résolution) et explicites (👍/👎).
2. Dashboard de suivi des métriques agrégées (TruLens ou Phoenix).
3. Alerter sur les dérives : baisse de Faithfulness, augmentation du taux de 👎.

---

# PARTIE 2 — Étude du besoin métier et usages cibles

---

## 2.1 — Cartographie des usages à court terme (tickets de support)

### 2.1.1 — Utilisateurs cibles

| Profil | Rôle | Fréquence d'usage | Attentes |
|--------|------|-------------------|----------|
| **Agent support N1** | Traite les tickets entrants, cherche des résolutions connues | Très fréquent (20–50 requêtes/jour) | Réponse rapide (<2s), tickets similaires déjà résolus, step-by-step de résolution |
| **Agent support N2/N3** | Traite les escalades, problèmes complexes | Fréquent (5–15 requêtes/jour) | Historique complet, documentation technique associée, tickets multi-produits |
| **Chef de projet / Team Lead** | Suivi des tendances, analyse des récurrences | Hebdomadaire | Synthèse par catégorie, produit, client. Métriques de volume et récurrences |
| **Manager / Direction** | Pilotage de la qualité de service | Mensuel | Tableaux de bord, KPIs, rapports d'analyse |
| **Consultant / Avant-vente** | Prépare des réponses à des appels d'offres, capitalise sur les retours clients | Ponctuel | Recherche par produit/client, synthèse des problèmes connus |

### 2.1.2 — Besoins de recherche identifiés

| Type de recherche | Description | Exemple de requête |
|-------------------|-------------|-------------------|
| **Par symptôme / problème** | L'utilisateur décrit un problème et cherche des résolutions | « L'imprimante X ne se connecte plus en WiFi après mise à jour firmware » |
| **Par client** | Retrouver l'historique support d'un client spécifique | « Tous les tickets du client Acme Corp des 6 derniers mois » |
| **Par produit** | Retrouver les incidents liés à un produit ou composant | « Problèmes connus sur le module de facturation v3.2 » |
| **Par solution** | Chercher une procédure de résolution connue | « Comment reconfigurer le connecteur LDAP après changement de certificat ? » |
| **Par catégorie** | Filtrer par type d'incident (bug, feature request, config, etc.) | « Tous les bugs critiques sur l'API REST » |
| **Recherche sémantique libre** | Question en langage naturel | « Quand un client se plaint de lenteurs, quelles sont les résolutions les plus fréquentes ? » |
| **Analyse de tendance** | Identification de patterns récurrents | « Quels sont les 5 problèmes les plus fréquents ce trimestre ? » |

### 2.1.3 — Formats de réponse attendus

| Format | Cas d'usage | Implémentation |
|--------|------------|----------------|
| **Liste de tickets similaires classés par pertinence** | Agent N1 cherchant un précédent | Résultats retrieval avec score, snippet, lien vers le ticket source |
| **Résumé de résolution** | Agent N1/N2 cherchant une solution rapide | LLM synthétise les résolutions des tickets similaires en un paragraphe actionnable |
| **Step-by-step de résolution** | Agent N1 appliquant une procédure | LLM restructure la résolution en étapes numérotées |
| **Fiche d'analyse** | Chef de projet, manager | Agrégation : nombre de tickets, répartition par sévérité, temps de résolution moyen |
| **Comparaison** | Agent N2/N3 évaluant plusieurs pistes | Tableau comparatif de plusieurs résolutions possibles |

### 2.1.4 — Latence acceptable

| Profil | Latence max acceptable | Justification |
|--------|----------------------|---------------|
| Agent support (en appel) | **< 2 secondes** | Le client est en ligne, chaque seconde compte |
| Agent support (traitement batch) | **< 5 secondes** | Traitement différé, tolérance plus grande |
| Manager / analyste | **< 10 secondes** | Requêtes complexes (agrégation), résultats riches |

**Implications techniques** :
- Le retrieval pur (hybrid search + reranking) doit rester sous 500 ms.
- La génération LLM (optionnelle dans le MVP) ajoute 1–3s selon le modèle et la longueur de la réponse.
- Le MVP sans LLM (retrieval only) respecte facilement la contrainte de 2s.

### 2.1.5 — Contraintes de confidentialité

| Contrainte | Description | Implémentation |
|------------|-------------|----------------|
| **Accès par rôle (RBAC)** | Un agent N1 ne voit que les tickets de son périmètre produit/client | Filtre métadonnées sur `client_id`, `product`, `team` dans les requêtes OpenSearch |
| **Cloisonnement client** | Les tickets d'un client ne doivent pas être suggérés lors du support d'un autre client (sauf si anonymisés) | Filtre strict sur `client_id` ; variante : anonymisation des chunks (remplacement du `client_id` par un placeholder) pour le mode « base de connaissances globale » |
| **PII (données personnelles)** | Emails, noms, téléphones, adresses dans les tickets | Pipeline de suppression/masquage PII à l'ingestion (regex + NER). Conformité RGPD. |
| **Audit trail** | Traçabilité de qui a cherché quoi, quand | Logging des requêtes avec user_id, timestamp, résultats retournés |
| **Droit à l'oubli** | Suppression des données d'un client sur demande (RGPD Art. 17) | Capacité de supprimer tous les chunks d'un `client_id` de l'index OpenSearch et de régénérer l'index |

---

## 2.2 — Extensions à moyen terme (ERP, CRM, PIM, GED, SAV)

### 2.2.1 — Analyse source par source

#### ERP (SAP, Odoo, etc.)

- **Nature des données** : hautement structurées (SQL). Commandes, lignes de commande, factures, bons de livraison, articles, stocks, contrats.
- **Volume** : important (millions de lignes transactionnelles). Cependant, seule une partie est pertinente pour le RAG (fiches articles, contrats, historique commandes par client).
- **Fréquence MAJ** : temps réel (transactions), mais les données structurelles (articles, contrats) changent moins fréquemment (quotidien à hebdomadaire).
- **Défis d'intégration** :
  - Les données ERP sont relationnelles → nécessitent une transformation en texte naturel (text-to-SQL inverse) ou en chunks structurés avant embedding.
  - Accès souvent restreint (VPN, firewall, autorisations SAP granulaires).
  - Schéma complexe (SAP : des milliers de tables avec des noms cryptiques — VBAK, MARA, KNA1).
- **Valeur ajoutée RAG** : « Un agent support peut demander 'Quel est le statut de la commande 12345 ?' et obtenir une réponse contextualisée sans naviguer dans l'ERP ».

#### CRM (Salesforce, HubSpot)

- **Nature des données** : semi-structurées. Fiches clients, contacts, opportunités commerciales, historique d'interactions (emails, appels, notes), contrats.
- **Volume** : modéré (milliers à dizaines de milliers de comptes, dizaines de milliers d'interactions).
- **Fréquence MAJ** : quotidienne (nouvelles interactions, mises à jour de statut).
- **Défis d'intégration** :
  - APIs REST bien documentées (Salesforce REST API, HubSpot API).
  - Les notes de contact et emails contiennent du texte libre riche → bon candidat pour embedding.
  - Gestion des doublons (même client sous différents noms).
- **Valeur ajoutée RAG** : « Avant de traiter un ticket, l'agent voit automatiquement le contexte client (contrat, historique d'achats, interactions commerciales récentes) ».

#### PIM (Akeneo, etc.)

- **Nature des données** : structurées (attributs produit : nom, référence, caractéristiques techniques) + médias (images, PDF de fiches techniques).
- **Volume** : modéré (milliers à dizaines de milliers de fiches produits).
- **Fréquence MAJ** : faible à modérée (nouvelles versions produit, ajout de produits).
- **Défis d'intégration** :
  - API REST Akeneo bien documentée.
  - Les attributs structurés doivent être sérialisés en texte naturel pour l'embedding.
  - Les fiches techniques PDF nécessitent une extraction OCR/texte.
- **Valeur ajoutée RAG** : « L'agent support peut demander 'Quelle est la configuration minimale requise pour le produit X ?' et obtenir la réponse directement à partir de la fiche PIM ».

#### GED (SharePoint, Confluence, Google Drive)

- **Nature des données** : non structurées. PDFs, documents Word, wikis, pages Confluence, présentations.
- **Volume** : potentiellement très important (des milliers de documents, certains volumineux).
- **Fréquence MAJ** : variable (documents vivants vs. archives).
- **Défis d'intégration** :
  - **Extraction de texte** : PDF (pypdf, pdfplumber, Unstructured), Word (python-docx), Confluence (API REST + HTML parsing).
  - **Qualité d'extraction** : les PDFs scannés nécessitent de l'OCR (Tesseract, Azure Document Intelligence). Les tableaux et schémas sont difficiles à extraire.
  - **Versioning** : un même document peut avoir plusieurs versions → gérer l'indexation/désindexation.
  - **Permissions** : les GED ont souvent des ACL complexes à respecter.
- **Valeur ajoutée RAG** : « L'agent accède à la documentation technique (guides d'installation, release notes, FAQ internes) directement depuis l'interface de recherche ».

#### SAV / Ticketing (Jira, Zendesk, Freshdesk)

- **Nature des données** : semi-structurées. Titre, description, commentaires, statut, priorité, catégorie, tags, assignee, timestamps + texte libre (descriptions, conversations).
- **Volume** : pour le MVP, ~200k tickets (dataset Kaggle). En entreprise, des centaines de milliers à millions de tickets historiques.
- **Fréquence MAJ** : temps réel (nouveaux tickets, mises à jour de statut, commentaires).
- **Défis d'intégration** :
  - APIs REST matures (Zendesk, Jira Service Management).
  - Volume de texte important (un ticket peut avoir 10–50 commentaires).
  - Bruit : signatures email, templates automatiques, notifications système à filtrer.
- **Valeur ajoutée RAG** : source primaire du MVP. Recherche de tickets similaires, capitalisation des résolutions.

### 2.2.2 — Tableau comparatif des sources

| Source | Type de données | Volume estimé (entreprise B2B mid-size) | Fréquence MAJ | Enjeux RAG spécifiques |
|--------|----------------|----------------------------------------|---------------|----------------------|
| **ERP** | Structuré (SQL) | 1–10M lignes transactionnelles ; ~10k fiches articles | Temps réel (transactions), quotidien (réf.) | Transformation SQL → texte, accès restreint, schéma complexe |
| **CRM** | Semi-structuré (JSON/API + texte libre) | 5–50k comptes, 100k+ interactions | Quotidien | Déduplication, notes texte + emails, sensible RGPD (données personnelles) |
| **PIM** | Structuré + médias (JSON + PDF/images) | 1–20k fiches produits | Hebdomadaire à mensuel | Sérialisation attributs → texte, extraction PDF, multimédia |
| **GED** | Non structuré (PDF, docx, wiki) | 5–50k documents, taille totale 1–100 Go | Variable (documents vivants) | Extraction texte, OCR, versioning, ACL, qualité variable |
| **SAV/Ticketing** | Semi-structuré (JSON + texte libre) | 50k–500k tickets | Temps réel | Bruit (signatures, templates), volume de texte, historique conversationnel |

---

## 2.3 — Risques identifiés

### 2.3.1 — Risques techniques

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| **Qualité d'indexation insuffisante** : les embeddings ne capturent pas le jargon métier | Moyenne | Élevé | Fine-tuning du modèle d'embedding, enrichissement des chunks avec métadonnées, hybrid search (BM25 capture le lexique exact) |
| **Hallucinations du LLM** : réponses factuellement incorrectes | Élevée | Critique | Faithfulness monitoring (RAGAs), affichage des sources, mode « retrieval only » (MVP sans génération), guardrails sur le prompt |
| **Dérive des embeddings** : nouveaux produits/termes mal représentés | Moyenne | Moyen | Re-indexation périodique, monitoring de la distribution des scores de similarité, alerte sur les requêtes à faible score max |
| **Mauvaise segmentation (chunking)** : chunks trop longs ou trop courts, découpage cassant la cohérence | Moyenne | Élevé | Tests A/B sur les stratégies de chunking (fixed-size vs. semantic), évaluation Recall@k sur un jeu de test |
| **Scalabilité OpenSearch** : performances dégradées avec la croissance du volume | Faible (MVP) | Moyen | Sizing initial approprié, monitoring des latences p95/p99, sharding horizontal |
| **Latence excessive** : pipeline RAG trop lent pour un usage en temps réel | Moyenne | Élevé | Optimisation par étapes : caching des requêtes fréquentes, batch embedding, reranking sur top-20 (et non top-100) |

### 2.3.2 — Risques métier

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| **Désintermédiation des experts** : les agents support cessent de développer leur expertise, se reposant aveuglément sur le RAG | Moyenne | Moyen | Former les agents : le RAG est un assistant, pas un oracle. Afficher les scores de confiance. |
| **Confiance excessive** : les utilisateurs font confiance aux réponses sans vérifier les sources | Élevée | Critique | Toujours afficher le lien vers le ticket source, exiger une validation humaine pour les résolutions critiques |
| **Adoption faible** : les utilisateurs trouvent l'outil plus lent ou moins intuitif que leur méthode actuelle | Moyenne | Élevé | UX soignée, temps de réponse < 2s, accompagnement au changement, quick wins visibles |
| **Biais de confirmation** : le système renforce des patterns de résolution sous-optimaux (car historiquement fréquents) | Faible | Moyen | Diversité des résultats (ne pas se limiter au cluster le plus similaire), revue périodique des auto-résolutions |

### 2.3.3 — Risques de gouvernance

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| **Non-conformité RGPD** : données personnelles dans les tickets indexés | Élevée | Critique | Pipeline PII de-identification à l'ingestion, DPA avec OpenRouter si API, DPIA formelle avant mise en production |
| **Violation d'accès** : un utilisateur accède à des tickets hors de son périmètre | Moyenne | Critique | RBAC strict avec filtrage par métadonnées, audit trail, tests de pénétration |
| **Traçabilité insuffisante** : impossible de reconstituer pourquoi une réponse a été donnée | Moyenne | Élevé | Logging exhaustif (question, chunks récupérés avec scores, réponse générée, modèle utilisé, timestamp) |
| **Dépendance à un fournisseur (vendor lock-in)** : OpenRouter / modèle API | Moyenne | Moyen | Abstraction de la couche embedding/LLM, capacité de basculer vers un modèle auto-hébergé |

### 2.3.4 — Risques de déploiement

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| **Intégration SI complexe** : connecteurs non disponibles, APIs instables | Élevée | Élevé | Commencer par le ticketing (API mature), étendre progressivement. Utiliser un ETL standard (Airbyte) pour standardiser. |
| **Maintenabilité** : le pipeline RAG devient un « plat de spaghetti » | Moyenne | Élevé | Architecture modulaire (cf. RAG modulaire), documentation du pipeline, monitoring ops (latence, erreurs, volumes) |
| **Compétences** : l'équipe interne n'a pas les compétences ML/NLP pour maintenir le système | Élevée | Élevé | Formation, documentation opérationnelle détaillée, choix de technologies accessibles (OpenSearch, Python/FastAPI) |
| **Coût opérationnel** : les appels API LLM/embedding en production coûtent plus que prévu | Moyenne | Moyen | Monitoring des coûts API, caching des embeddings, modèles locaux en fallback |

---

# PARTIE 3 — Étude du système d'information existant

---

## 3.1 — Cartographie du SI supposé

Pour une entreprise B2B technologique de taille intermédiaire (100–1000 employés, activité logicielle ou matérielle), le SI typique se compose des briques suivantes :

### 3.1.1 — Ticketing / Service Management

| Outil | Description | Accès données | Format |
|-------|------------|---------------|--------|
| **Jira Service Management** (Atlassian) | Gestion de tickets ITSM, SLA, files d'attente, automatisations | API REST v3 (OAuth 2.0, API Key) | JSON |
| **Zendesk** | Support client multicanal (email, chat, téléphone), knowledge base | API REST v2 (Bearer token) | JSON |
| **Freshdesk** (Freshworks) | Alternative SaaS au ticketing, orientée PME/ETI | API REST v2 (API Key) | JSON |

**Données pertinentes pour le RAG** :
- Titre du ticket, description, commentaires/conversations
- Statut, priorité, catégorie, tags
- Assignee, reporter, date de création, date de résolution
- Résolution (champ texte libre ou structuré)
- Pièces jointes (PDF, captures d'écran)

**Volumes typiques** : 500–5000 tickets/mois pour une entreprise B2B mid-size → 50k–500k tickets sur 5–10 ans d'historique.

### 3.1.2 — ERP

| Outil | Description | Accès données | Format |
|-------|------------|---------------|--------|
| **SAP S/4HANA** | ERP leader pour les grandes entreprises. Modules : FI (finance), MM (achats), SD (ventes), PP (production). | OData API, RFC/BAPI, base HANA directe | JSON/XML/SQL |
| **Odoo** | ERP open source modulaire. Modules : CRM, ventes, achats, inventaire, comptabilité, fabrication. | JSON-RPC / XML-RPC API, accès PostgreSQL direct | JSON |

**Données pertinentes pour le RAG** :
- Fiches articles (références produit, descriptions, caractéristiques techniques)
- Contrats clients (dates, conditions, SLA associés)
- Historique commandes par client (produits achetés, dates, montants)
- Données logistiques (livraisons, retours)

### 3.1.3 — CRM

| Outil | Description | Accès données | Format |
|-------|------------|---------------|--------|
| **Salesforce** | CRM leader. Objets : Account, Contact, Opportunity, Case, Activity. | REST API / SOQL / Bulk API (OAuth 2.0) | JSON |
| **HubSpot** | CRM orienté marketing/ventes. Objets : Companies, Contacts, Deals, Tickets. | REST API (API Key ou OAuth 2.0) | JSON |

**Données pertinentes pour le RAG** :
- Fiches clients (nom, secteur, taille, interlocuteurs)
- Historique des interactions commerciales (notes d'appels, emails)
- Opportunités en cours (contexte commercial lors du support)
- Contrats et engagements SLA

### 3.1.4 — PIM

| Outil | Description | Accès données | Format |
|-------|------------|---------------|--------|
| **Akeneo** | PIM open source de référence. Gestion centralisée des catalogues produits. | REST API (OAuth 2.0) | JSON |

**Données pertinentes pour le RAG** :
- Catalogue produits : noms, descriptions, caractéristiques techniques, familles
- Attributs multi-langues (fr/en)
- Médias associés (images, fiches techniques PDF)
- Associations entre produits (accessoires, remplacements, upsells)

### 3.1.5 — GED / Documentation

| Outil | Description | Accès données | Format |
|-------|------------|---------------|--------|
| **SharePoint** (Microsoft 365) | GED d'entreprise, stockage de documents, sites d'équipe. | Microsoft Graph API (OAuth 2.0) | Fichiers binaires + métadonnées JSON |
| **Confluence** (Atlassian) | Wiki d'entreprise, documentation technique, base de connaissances interne. | REST API v2 (OAuth 2.0, API Key) | HTML/JSON |
| **Google Drive** / Google Workspace | Stockage cloud, documents collaboratifs. | Google Drive API v3 (OAuth 2.0) | Fichiers + métadonnées JSON |

**Données pertinentes pour le RAG** :
- Documentation technique (guides d'installation, release notes, architecture)
- Procédures internes (runbooks, playbooks de support)
- FAQ internes
- Présentations commerciales (contexte produit)

---

## 3.2 — Stratégies d'intégration

### 3.2.1 — On-premise vs. Cloud vs. Hybride

| Stratégie | Description | Avantages | Inconvénients | Quand la choisir |
|-----------|------------|-----------|---------------|------------------|
| **Full cloud** | OpenSearch managé (AWS OpenSearch Service, Aiven), modèles via API (OpenRouter), backend déployé sur cloud (AWS, GCP, Azure) | Pas d'infra à gérer, scalabilité élastique, time-to-market rapide | Coût récurrent, données hébergées chez un tiers, latence réseau | MVP, entreprises cloud-native, pas de contrainte forte de souveraineté |
| **On-premise** | OpenSearch auto-hébergé (Docker/K8s on-prem), modèles d'embedding locaux, GPU on-prem | Contrôle total des données, pas de coût API variable, latence minimale | Investissement matériel (serveurs GPU), maintenance opérationnelle, recrutement DevOps/ML | Secteurs réglementés (défense, santé), entreprises avec infra existante |
| **Hybride** | OpenSearch on-prem ou cloud privé + API externes (OpenRouter) pour les LLMs/embeddings, avec anonymisation des données avant envoi | Compromis coût/contrôle, flexibilité | Complexité d'architecture, double gestion | Majorité des entreprises B2B — approche recommandée pour RAG-Time |

### 3.2.2 — Connecteurs et ETL

| Approche | Outils | Avantages | Inconvénients |
|----------|--------|-----------|---------------|
| **Connecteurs natifs (custom)** | Scripts Python (requests, zeep), SDKs officiels (Salesforce simple_salesforce, Jira jira-python) | Flexibilité totale, maîtrise du code | Maintenance à charge de l'équipe, pas de gestion native des erreurs/retry |
| **ETL / iPaaS** | **Airbyte** (open source, 300+ connecteurs), **n8n** (workflow automation), **Fivetran** (SaaS), **Apache NiFi** (on-prem) | Connecteurs pré-construits, gestion des erreurs, monitoring, schéma de destination configurable | Couche supplémentaire, courbe d'apprentissage, certains connecteurs limités |
| **CDC (Change Data Capture)** | **Debezium** (PostgreSQL/MySQL), **DynamoDB Streams** | Temps réel, capture les changements incrémentaux directement depuis la base | Complexité opérationnelle, nécessite un broker (Kafka) |

**Recommandation RAG-Time** :
- **MVP** : connecteur Python custom pour le ticketing (API Zendesk/Jira → script d'ingestion). Simple et rapide à développer.
- **Production / multi-sources** : **Airbyte** comme hub d'intégration. Il dispose de connecteurs natifs pour Zendesk, Jira, Salesforce, HubSpot, SharePoint, PostgreSQL (Odoo). Les données sont extraites vers un staging (data lake ou base intermédiaire), puis un pipeline Python les transforme en chunks et les indexe dans OpenSearch.

### 3.2.3 — Synchronisation : temps réel vs. batch

| Mode | Mécanisme | Latence | Cas d'usage |
|------|-----------|---------|-------------|
| **Batch (scheduled)** | Cron job nocturne ou toutes les X heures. Le connecteur extrait tous les documents modifiés depuis le dernier run (delta sync basé sur `updated_at`). | Minutes à heures | Données peu volatiles (documentation, fiches produits, historique ERP). Suffisant pour le MVP. |
| **Quasi-temps réel** | **Webhooks** : la source (Zendesk, Jira) envoie un événement HTTP à chaque création/modification de ticket. Le pipeline RAG traite l'événement et met à jour l'index. | Secondes à minutes | Tickets de support (un nouveau ticket doit être indexé rapidement pour être retrouvé par d'autres agents). |
| **Temps réel (streaming)** | **CDC** (Debezium + Kafka) ou event bus (RabbitMQ, AWS SQS). | Sous la seconde | Volume très élevé + exigence de fraîcheur (pas nécessaire pour le MVP). |

**Architecture recommandée RAG-Time** :
- Tickets : **webhooks** (Zendesk/Jira → endpoint FastAPI → pipeline d'indexation → OpenSearch)
- Autres sources (ERP, CRM, PIM, GED) : **batch nocturne** (Airbyte scheduled sync → staging → pipeline d'indexation)

### 3.2.4 — Gestion du changement de schéma et versioning

| Enjeu | Description | Solution |
|-------|-------------|----------|
| **Changement de schéma source** | Un champ est renommé, supprimé ou ajouté dans l'API source (ex. : Zendesk change le format de la réponse). | Versionner les connecteurs, tests d'intégration automatisés, alertes sur les erreurs d'ingestion. |
| **Versioning des chunks** | Quand un document est mis à jour, faut-il supprimer l'ancien chunk et le remplacer, ou garder l'historique ? | Chaque chunk a un `doc_id` + `version`. À chaque mise à jour, l'ancien chunk est marqué obsolète (`is_current: false`) et le nouveau est indexé. Les recherches filtrent sur `is_current: true` par défaut. |
| **Versioning du modèle d'embedding** | Changer de modèle d'embedding nécessite de ré-encoder tout le corpus (les vecteurs ne sont pas comparables entre modèles). | Stocker le `model_version` avec chaque chunk. Lors d'un changement de modèle, ré-indexer tout le corpus (blue-green deployment : nouvel index en parallèle, bascule quand prêt). |
| **Migration d'index OpenSearch** | Changement de mapping (ex. : ajout d'un champ, changement de dimension d'embedding). | Utiliser les alias OpenSearch pour pointer vers l'index actif, créer le nouvel index, réindexer (reindex API), basculer l'alias. |

---

## 3.3 — Architecture de collecte recommandée

### 3.3.1 — Vue d'ensemble du pipeline d'ingestion

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SOURCES DE DONNÉES                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Ticketing│  │   ERP    │  │   CRM    │  │   PIM    │  │   GED    │  │
│  │(Zendesk) │  │  (Odoo)  │  │(HubSpot) │  │(Akeneo)  │  │(Conflce) │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │webhook       │batch       │batch        │batch        │batch    │
└───────┼──────────────┼────────────┼─────────────┼─────────────┼─────────┘
        │              │            │             │             │
        ▼              ▼            ▼             ▼             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     COUCHE D'INTÉGRATION                                 │
│                                                                          │
│  ┌────────────────────────┐    ┌────────────────────────────────────┐    │
│  │  Webhook Receiver      │    │  Airbyte (batch connector)         │    │
│  │  (FastAPI endpoint)    │    │  Scheduled sync (nightly)          │    │
│  │  → tickets temps réel  │    │  → ERP, CRM, PIM, GED             │    │
│  └──────────┬─────────────┘    └──────────────┬─────────────────────┘    │
│             │                                  │                         │
│             ▼                                  ▼                         │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    STAGING (Raw Data Store)                      │    │
│  │              PostgreSQL / MinIO / fichiers JSON                  │    │
│  └──────────────────────────────┬───────────────────────────────────┘    │
└─────────────────────────────────┼────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE DE TRANSFORMATION                             │
│                                                                          │
│  1. NETTOYAGE           2. EXTRACTION         3. SEGMENTATION           │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐       │
│  │• Suppr. HTML  │      │• Métadonnées │      │• Chunking         │       │
│  │• Suppr. PII   │  →   │  (client_id, │  →   │  sémantique       │       │
│  │• Normalisation│      │  category,   │      │• 400 tokens       │       │
│  │  unicode      │      │  product)    │      │• Overlap 50       │       │
│  │• Déduplication│      │• OCR (PDFs)  │      │• Métadonnées      │       │
│  └──────────────┘      └──────────────┘      │  héritées         │       │
│                                               └────────┬─────────┘       │
│                                                        │                 │
│  4. EMBEDDING                    5. INDEXATION                           │
│  ┌─────────────────────┐        ┌─────────────────────────────┐         │
│  │ BGE-M3 / OpenRouter  │   →   │ OpenSearch                   │         │
│  │ → vecteur 1024 dims  │       │ • Index BM25 (text)          │         │
│  │ Normalisation L2     │       │ • Index k-NN (HNSW, cosine)  │         │
│  └─────────────────────┘        │ • Métadonnées (keyword/date) │         │
│                                  └─────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3.2 — Patterns d'ingestion par source

#### Ticketing (MVP — source primaire)

```python
# Pattern simplifié — ingestion webhook ticket
# 1. Recevoir le webhook Zendesk/Jira
# 2. Extraire les champs pertinents
# 3. Nettoyer, segmenter, embedder, indexer

# Endpoint FastAPI
@app.post("/webhook/ticket")
async def ingest_ticket(payload: TicketWebhook):
    # Extraction
    ticket = extract_ticket_fields(payload)  # {id, title, description, resolution, category, client_id, ...}
    
    # Nettoyage
    ticket.description = clean_html(ticket.description)
    ticket.description = remove_pii(ticket.description)
    
    # Segmentation
    chunks = semantic_chunker(
        text=f"{ticket.title}\n\n{ticket.description}\n\nResolution:\n{ticket.resolution}",
        chunk_size=400,
        overlap=50
    )
    
    # Enrichissement métadonnées
    for chunk in chunks:
        chunk.metadata = {
            "source": "ticketing",
            "ticket_id": ticket.id,
            "client_id": ticket.client_id,
            "category": ticket.category,
            "product": ticket.product,
            "created_at": ticket.created_at,
            "is_current": True
        }
    
    # Embedding + indexation
    embeddings = embed_model.encode([c.text for c in chunks])
    opensearch_bulk_index(chunks, embeddings)
```

#### ERP (batch)

```
Odoo PostgreSQL → Airbyte (nightly) → Staging JSON
→ Script Python : sérialisation des fiches articles en texte naturel
   ("Produit : {nom}, Référence : {ref}, Description : {desc}, Prix : {prix}...")
→ Chunking → Embedding → OpenSearch
```

#### CRM (batch)

```
Salesforce API → Airbyte (nightly) → Staging JSON
→ Script Python : concaténation fiche client + dernières interactions
   ("Client : {nom}, Secteur : {secteur}, Contrat : {type}...\nDernières interactions :\n{notes}")
→ Chunking → Embedding → OpenSearch
```

#### PIM (batch)

```
Akeneo API → Airbyte (nightly) → Staging JSON
→ Script Python : sérialisation fiche produit + extraction texte des PDF techniques
→ Chunking → Embedding → OpenSearch
```

#### GED (batch)

```
Confluence API / SharePoint Graph API → Airbyte (nightly) → Staging (fichiers)
→ Extraction texte (HTML → markdownify, PDF → pypdf/pdfplumber, DOCX → python-docx)
→ Nettoyage, chunking → Embedding → OpenSearch
```

### 3.3.3 — Orchestration et monitoring

| Composant | Outil recommandé | Rôle |
|-----------|-----------------|------|
| **Orchestration batch** | Apache Airflow / Prefect / simple cron | Planification des jobs d'ingestion nocturnes, gestion des dépendances, retry |
| **Queue de messages** | Redis / RabbitMQ | Buffer entre les webhooks et le pipeline de traitement (découplage, gestion de la charge) |
| **Monitoring pipeline** | Prometheus + Grafana | Métriques : nombre de documents indexés, temps d'ingestion, taux d'erreur, latence d'embedding |
| **Monitoring OpenSearch** | OpenSearch Dashboards | Santé du cluster, taille des index, latence des requêtes, utilisation mémoire |
| **Logging & tracing** | ELK (OpenSearch + Logstash + Dashboards) ou OpenTelemetry | Traçabilité de chaque document depuis la source jusqu'à l'index |
| **Alerting** | Grafana Alerts / PagerDuty | Alertes sur : ingestion en erreur, index indisponible, latence anormale |

---

# ANNEXES

## Glossaire

| Terme | Définition |
|-------|-----------|
| **ANN** | Approximate Nearest Neighbors — algorithme de recherche approximative de plus proches voisins |
| **BM25** | Best Matching 25 — algorithme de scoring full-text basé sur TF-IDF avec saturation et normalisation de longueur |
| **Chunk** | Fragment de document de taille calibrée, unité de base de l'indexation RAG |
| **Cross-encoder** | Modèle qui prend en entrée la concaténation (question, document) et produit un score de pertinence |
| **Embedding** | Vecteur numérique dense représentant un texte dans un espace sémantique |
| **HNSW** | Hierarchical Navigable Small World — algorithme de graphe pour la recherche ANN |
| **HyDE** | Hypothetical Document Embeddings — technique de pré-retrieval |
| **k-NN** | k-Nearest Neighbors — recherche des k plus proches voisins dans un espace vectoriel |
| **LLM** | Large Language Model — modèle de langage de grande taille |
| **MTEB** | Massive Text Embedding Benchmark — benchmark de référence pour les modèles d'embedding |
| **PII** | Personally Identifiable Information — données personnelles identifiantes |
| **RAG** | Retrieval-Augmented Generation — paradigme combinant recherche et génération |
| **RBAC** | Role-Based Access Control — contrôle d'accès par rôle |
| **RRF** | Reciprocal Rank Fusion — méthode de fusion de classements |

## Références bibliographiques

1. Asai, A. et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. arXiv:2310.11511.
2. Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216.
3. Cormack, G.V. et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. SIGIR.
4. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805.
5. Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130.
6. Es, S. et al. (2023). *RAGAs: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.
7. Fan, W. et al. (2025). *A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions*. arXiv:2506.00054.
8. Gao, L. et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. arXiv:2212.10496.
9. Gao, Y. et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
10. Johnson, J. et al. (2019). *Billion-scale similarity search with GPUs (FAISS)*. arXiv:1702.08734.
11. Khattab, O. et al. (2022). *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*. arXiv:2310.03714.
12. Lee, N. et al. (2024). *Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?*. arXiv:2406.13121.
13. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401.
14. Liu, N.F. et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. arXiv:2307.03172.
15. Ma, X. et al. (2023). *Query Rewriting for Retrieval-Augmented Large Language Models*. arXiv:2305.14283.
16. Malkov, Y.A. & Yashunin, D.A. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. arXiv:1603.09320.
17. Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781.
18. Muennighoff, N. et al. (2023). *MTEB: Massive Text Embedding Benchmark*. arXiv:2210.07316.
19. Pan, S. et al. (2024). *Unifying Large Language Models and Knowledge Graphs: A Roadmap*. arXiv:2306.08302.
20. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.
21. Saad-Falcon, J. et al. (2024). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*. arXiv:2311.09476.
22. Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761.
23. Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. arXiv:2104.08663.
24. Vaswani, A. et al. (2017). *Attention Is All You Need*. arXiv:1706.03762.
25. Wang, L. et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. arXiv:2212.03533.
26. Xu, W. et al. (2024). *Retrieval Head Mechanistically Explains Long-Context Factuality*. arXiv:2404.15574.
27. Yan, S.-Q. et al. (2024). *Corrective Retrieval Augmented Generation (CRAG)*. arXiv:2401.15884.
28. Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.
29. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685.
30. Zhao, P. et al. (2024). *Retrieval-Augmented Generation for AI-Generated Content: A Survey*. arXiv:2402.19473.

## Ressources complémentaires

- OpenSearch Documentation — Vector Search : https://docs.opensearch.org/latest/vector-search/
- OpenSearch Documentation — Semantic Search : https://docs.opensearch.org/latest/vector-search/ai-search/semantic-search/
- RAG Survey GitHub : https://github.com/hymie122/RAG-Survey
- Dataset MVP (Kaggle, 200k tickets) : https://www.kaggle.com/datasets/mirzayasirabdullah07/customer-support-tickets-dataset-200k-records
- Mistral AI — LLM as RAG Judge : https://mistral.ai/news/llm-as-rag-judge
- Hugging Face MTEB Leaderboard : https://huggingface.co/spaces/mteb/leaderboard
- LangChain Documentation : https://python.langchain.com/
- LlamaIndex Documentation : https://docs.llamaindex.ai/
