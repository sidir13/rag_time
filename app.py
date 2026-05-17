"""Frontend Streamlit — RAG-Time
================================

Interface de chat RAG pour tickets de support technique.

Fonctionnalités :
- Indexation du CSV de tickets (200k) avec limite configurable
- Upload de documents additionnels (PDF, TXT, MD)
- 10 filtres métadonnées (product, category, priority, status, etc.)
- Chat avec affichage des sources et scores de similarité
- Mode dégradé (retrieval-only) si ANTHROPIC_API_KEY absente

Lancement : streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ingest import Ingestor, DATA_RAW, VECTORSTORE_DIR  # noqa: E402
from rag import RAGEngine  # noqa: E402

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# Champs de filtrage avec labels français pour la sidebar
FILTER_FIELDS = {
    "product": "Produit",
    "category": "Catégorie",
    "priority": "Priorité",
    "status": "Statut",
    "channel": "Canal",
    "region": "Région",
    "language": "Langue",
    "operating_system": "OS",
    "subscription_type": "Abonnement",
    "customer_segment": "Segment",
}


# ── Chargement des options de filtres (depuis le CSV) ────────────────────────

@st.cache_data
def get_filter_options() -> dict[str, list[str]]:
    """Charge les valeurs uniques pour les dropdowns de filtres."""
    if not DATA_RAW.exists():
        return {}
    try:
        header = pd.read_csv(DATA_RAW, nrows=0).columns.tolist()
        cols = [c for c in FILTER_FIELDS if c in header]
        if not cols:
            return {}
        df = pd.read_csv(DATA_RAW, usecols=cols)
        return {
            field: sorted(df[field].dropna().unique().astype(str).tolist())
            for field in cols
        }
    except Exception:
        return {}


# ── Initialisation système (singleton — modèle + index FAISS partagés) ────────

@st.cache_resource
def init_system():
    """Initialise le modèle d'embedding, l'index FAISS, Ingestor et RAGEngine.

    Un seul modèle BGE-M3 (~2.3 Go) et un seul index FAISS sont partagés
    entre l'ingesteur et le moteur RAG pour éviter la duplication en mémoire.
    """
    ingestor = Ingestor()
    rag_engine = RAGEngine(
        model=ingestor.model,
        index=ingestor.index,
        store=ingestor.store,
    )
    return ingestor, rag_engine


# ── Configuration page ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG-Time | Support Tickets",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chargement du système
with st.spinner(
    "Chargement du modèle BGE-M3 "
    "(le premier lancement télécharge ~2.3 Go)..."
):
    ingestor, rag_engine = init_system()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 RAG-Time")
    st.caption("RAG pour tickets de support · BGE-M3 + FAISS + GPT-4o")
    st.divider()

    # -- Indexation CSV --
    st.subheader("📥 Indexation")
    st.metric("Documents indexés", f"{ingestor.count:,}")

    with st.expander("Indexer le CSV de tickets", expanded=ingestor.count == 0):
        max_rows = st.number_input(
            "Limite de lignes (0 = tout le CSV)",
            min_value=0,
            max_value=200_000,
            value=1000,
            step=500,
            help=(
                "Pour le développement, commencez par 500-1000 tickets. "
                "L'indexation complète (200k) peut prendre plusieurs heures sur CPU."
            ),
        )
        resolved_only = st.checkbox(
            "Résolus uniquement",
            value=False,
            help="N'indexer que les tickets avec status Resolved ou Closed",
        )
        if st.button(
            "🚀 Lancer l'indexation", type="primary", use_container_width=True
        ):
            n_rows = max_rows if max_rows > 0 else None
            label = f"{max_rows}" if max_rows > 0 else "200 000"
            with st.spinner(f"Indexation de {label} tickets en cours..."):
                n = ingestor.ingest_csv(
                    resolved_only=resolved_only,
                    max_rows=n_rows,
                )
            st.success(f"✅ {n:,} tickets indexés")
            st.rerun()

    # -- Upload de documents --
    with st.expander("Uploader des documents"):
        uploaded = st.file_uploader(
            "PDF, TXT ou Markdown",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Indexer les documents"):
            for f in uploaded:
                with st.spinner(f"Indexation de {f.name}..."):
                    try:
                        suffix = Path(f.name).suffix
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(f.getvalue())
                            tmp_path = Path(tmp.name)
                        ingestor.ingest_document(tmp_path)
                        os.unlink(tmp_path)
                        st.success(f"✅ {f.name} indexé")
                    except Exception as e:
                        st.error(f"❌ {f.name} : {e}")
            st.rerun()

    # -- Filtres métadonnées --
    st.divider()
    st.subheader("🔎 Filtres de recherche")

    filter_options = get_filter_options()
    filters: dict[str, str] = {}

    for field, label in FILTER_FIELDS.items():
        choices = ["Tous"] + filter_options.get(field, [])
        value = st.selectbox(label, choices, key=f"filter_{field}")
        if value != "Tous":
            filters[field] = value

    top_k = st.slider("Nombre de résultats (top-k)", 1, 20, 5)

    # -- Actions --
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Vider l'index"):
            ingestor.clear()
            st.success("Index vidé")
            st.rerun()
    with col2:
        if st.button("💬 Nouveau chat"):
            st.session_state.messages = []
            st.rerun()

    # -- Statut API --
    st.divider()
    if rag_engine.has_llm:
        st.success("✅ API OpenRouter configurée")
    else:
        st.warning(
            "⚠️ OPENROUTER_API_KEY manquante\n\n"
            "Mode retrieval uniquement (pas de génération LLM)"
        )


# ── Zone principale — Chat ──────────────────────────────────────────────────

st.title("RAG-Time — Chat Support Technique")

if ingestor.count == 0:
    st.info(
        "👈 Commencez par indexer le CSV de tickets ou uploader des documents "
        "via la sidebar à gauche."
    )


def _render_sources(sources: list[dict]):
    """Affiche les sources dans un expander avec scores et métadonnées."""
    with st.expander(f"📎 Sources ({len(sources)} résultats)"):
        for s in sources:
            sim = f"{s['similarity']:.1%}"
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(
                    f"**{s['ticket_id']}** — "
                    f"{s.get('product', '')} · {s.get('category', '')}"
                )
                meta_parts = [
                    v
                    for k in ("language", "status", "priority", "region")
                    if (v := s.get(k))
                ]
                if meta_parts:
                    st.caption(" · ".join(meta_parts))
            with cols[1]:
                st.metric("Sim.", sim)
            st.code(s["excerpt"], language=None)
            st.divider()


# Historique du chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            _render_sources(msg["sources"])

# Saisie utilisateur
if prompt := st.chat_input("Posez votre question sur les tickets de support..."):
    # Message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse assistant
    with st.chat_message("assistant"):
        if ingestor.count == 0:
            st.warning("L'index est vide. Indexez d'abord des documents.")
            st.stop()

        with st.spinner("Recherche et génération en cours..."):
            result = rag_engine.query(
                question=prompt,
                top_k=top_k,
                filters=filters if filters else None,
            )

        st.markdown(result["answer"])

        if result["sources"]:
            _render_sources(result["sources"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )
