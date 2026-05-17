"""Test du pipeline RAG sur le dataset multilingual-customer-support-tickets
============================================================================

Étapes :
  1. Téléchargement du dataset Kaggle via kagglehub
  2. Exploration et mapping des colonnes vers le format attendu par le pipeline
  3. Indexation d'un échantillon (2 000 tickets) dans un vectorstore séparé
  4. Tests de requêtes RAG (retrieval + LLM)
  5. Évaluation automatique (10 questions, LLM-as-judge)
  6. Si résultats satisfaisants → relance avec GPT-4o

Usage :
    python test_multilingual_pipeline.py
    python test_multilingual_pipeline.py --sample-size 500 --n-eval 5
    python test_multilingual_pipeline.py --gpt4o     # force GPT-4o directement
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 ne supporte pas → ✓ ⚠ etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SAMPLE_VECTORSTORE = ROOT / "data" / "vectorstore_multilingual_test"

# ── Seuils de satisfaction ────────────────────────────────────────────────────
# Si le score composite dépasse ce seuil → on relance avec GPT-4o
SATISFACTION_THRESHOLD = 0.55


# ── Mapping des colonnes du dataset multilingual vers le format du pipeline ──
# Le dataset Kaggle "multilingual-customer-support-tickets" contient des colonnes
# légèrement différentes. Ce dict les mappe vers les noms attendus par ingest.py.
COLUMN_MAP = {
    # Kaggle column → pipeline column
    "Ticket ID":            "ticket_id",
    "ticket_id":            "ticket_id",
    "Customer Issue":       "issue_description",
    "Issue Description":    "issue_description",
    "issue":                "issue_description",
    "body":                 "issue_description",
    "text":                 "issue_description",
    "Resolution":           "resolution_notes",
    "Resolution Notes":     "resolution_notes",
    "resolution":           "resolution_notes",
    "response":             "resolution_notes",
    "answer":               "resolution_notes",
    "Answer":               "resolution_notes",
    "Product":              "product",
    "product":              "product",
    "Category":             "category",
    "category":             "category",
    "Type":                 "category",
    "Priority":             "priority",
    "priority":             "priority",
    "Status":               "status",
    "status":               "status",
    "Channel":              "channel",
    "channel":              "channel",
    "Language":             "language",
    "language":             "language",
    "lang":                 "language",
    "Region":               "region",
    "region":               "region",
    "Subject":              "subject",
    "subject":              "subject",
    "Customer Name":        "customer_name",
    "customer_name":        "customer_name",
    "Customer Email":       "customer_email",
    "customer_email":       "customer_email",
}


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes selon COLUMN_MAP (case-insensitive fallback)."""
    # Correspondance exacte d'abord
    rename = {col: COLUMN_MAP[col] for col in df.columns if col in COLUMN_MAP}
    df = df.rename(columns=rename)

    # Fallback case-insensitive pour les colonnes non encore mappées
    remaining = [c for c in df.columns if c not in rename.values()]
    lower_map = {k.lower(): v for k, v in COLUMN_MAP.items()}
    for col in remaining:
        if col.lower() in lower_map and lower_map[col.lower()] not in df.columns:
            df = df.rename(columns={col: lower_map[col.lower()]})

    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garantit la présence des colonnes minimales avec valeurs par défaut."""
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"ml_{i:06d}" for i in range(len(df))]

    if "issue_description" not in df.columns:
        # Essayer de concaténer subject + body si disponibles
        candidates = ["subject", "Subject", "title", "Title", "description", "Description"]
        found = [c for c in candidates if c in df.columns]
        if found:
            df["issue_description"] = df[found[0]].fillna("").astype(str)
            print(f"  ⚠  'issue_description' construite depuis '{found[0]}'")
        else:
            # Prendre la première colonne textuelle longue
            for col in df.columns:
                if df[col].dtype == object:
                    avg_len = df[col].dropna().astype(str).str.len().mean()
                    if avg_len and avg_len > 30:
                        df["issue_description"] = df[col].fillna("").astype(str)
                        print(f"  ⚠  'issue_description' construite depuis '{col}' (len moy={avg_len:.0f})")
                        break
            else:
                df["issue_description"] = "No description available"

    if "resolution_notes" not in df.columns:
        df["resolution_notes"] = ""

    return df


def download_dataset() -> Path:
    """Télécharge le dataset via kagglehub et retourne le chemin."""
    try:
        import kagglehub
    except ImportError:
        print("[!] kagglehub non installé → installation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub", "-q"])
        import kagglehub

    print("\n[1/5] Téléchargement du dataset Kaggle...")
    print("      tobiasbueck/multilingual-customer-support-tickets")
    path = kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets")
    print(f"      → Dataset téléchargé : {path}")
    return Path(path)


def load_sample(dataset_path: Path, sample_size: int = 2000) -> tuple[pd.DataFrame, Path]:
    """Charge et échantillonne le dataset, retourne le DataFrame et le CSV temporaire."""
    print(f"\n[2/5] Chargement et échantillonnage ({sample_size} tickets)...")

    # Chercher le(s) fichier(s) CSV dans le dossier téléchargé
    csv_files = list(dataset_path.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {dataset_path}")

    print(f"      Fichiers CSV disponibles :")
    for f in csv_files:
        size_mb = f.stat().st_size / 1_048_576
        print(f"        - {f.name}  ({size_mb:.1f} Mo)")

    # Choisir le plus gros CSV (le dataset principal)
    main_csv = max(csv_files, key=lambda f: f.stat().st_size)
    print(f"      → Fichier principal : {main_csv.name}")

    df = pd.read_csv(main_csv, low_memory=False)
    print(f"      Dimensions originales : {len(df):,} lignes × {len(df.columns)} colonnes")
    print(f"      Colonnes : {list(df.columns)}")

    # Mapping des colonnes
    df = _map_columns(df)
    df = _ensure_required_columns(df)

    # Échantillonnage stratifié par langue si possible
    if "language" in df.columns:
        langs = df["language"].value_counts()
        print(f"\n      Langues disponibles :")
        for lang, cnt in langs.head(10).items():
            print(f"        {lang}: {cnt:,}")

        # Stratified sample : max sample_size, proportionnel par langue
        n_langs = df["language"].nunique()
        per_lang = max(10, sample_size // n_langs)
        sampled = (
            df.groupby("language", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_lang), random_state=42))
            .reset_index(drop=True)
        )
        # Tronquer au total voulu
        sampled = sampled.sample(min(len(sampled), sample_size), random_state=42).reset_index(drop=True)
    else:
        sampled = df.sample(min(len(df), sample_size), random_state=42).reset_index(drop=True)

    print(f"\n      Échantillon : {len(sampled):,} tickets")
    if "language" in sampled.columns:
        print(f"      Distribution langues : {sampled['language'].value_counts().to_dict()}")

    # Sauvegarder le sample dans data/raw/
    sample_path = ROOT / "data" / "raw" / "multilingual_sample.csv"
    sampled.to_csv(sample_path, index=False)
    print(f"      → Sauvegardé : {sample_path}")

    return sampled, sample_path


def build_index(sample_csv: Path, use_gpt4o: bool = False, skip_index: bool = False) -> "RAGEngine":
    """Construit l'index FAISS et retourne le RAGEngine configuré."""
    from ingest import Ingestor
    from rag import RAGEngine

    print(f"\n[3/5] Indexation dans {SAMPLE_VECTORSTORE}...")

    # Injecter la clef API et le modèle d'embedding
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    # Forcer nomic-embed (768 dims, beaucoup plus rapide que bge-m3 sur CPU)
    os.environ["EMBED_MODEL"] = "nomic-ai/nomic-embed-text-v1.5"
    if use_gpt4o:
        os.environ["LLM_MODEL"] = "openai/gpt-4o"
        print("      Mode GPT-4o activé")
    else:
        os.environ["LLM_MODEL"] = "openai/gpt-4o-mini"
        print("      Mode GPT-4o-mini (test rapide)")

    if skip_index and SAMPLE_VECTORSTORE.exists():
        print("      --skip-index : vectorstore existant réutilisé")
        ingestor = Ingestor(
            persist_dir=SAMPLE_VECTORSTORE,
            embed_model="nomic-ai/nomic-embed-text-v1.5",
        )
        print(f"      → {ingestor.count:,} chunks déjà indexés")
    else:
        # Nettoyer l'ancien index de test si présent
        if SAMPLE_VECTORSTORE.exists():
            shutil.rmtree(SAMPLE_VECTORSTORE)
        SAMPLE_VECTORSTORE.mkdir(parents=True, exist_ok=True)

        # Passer embed_model explicitement pour ignorer tout .env qui forcerait bge-m3
        ingestor = Ingestor(
            persist_dir=SAMPLE_VECTORSTORE,
            embed_model="nomic-ai/nomic-embed-text-v1.5",
        )
        n_added = ingestor.ingest_csv(csv_path=sample_csv)
        print(f"      → {n_added:,} tickets indexés ({ingestor.count:,} chunks)")

    rag = RAGEngine(
        model=ingestor.model,
        index=ingestor.index,
        store=ingestor.store,
        persist_dir=SAMPLE_VECTORSTORE,
        embed_model="nomic-ai/nomic-embed-text-v1.5",
    )
    return rag


def run_smoke_tests(rag: "RAGEngine") -> None:
    """Lance 5 requêtes de smoke test multilingues."""
    print("\n[4/5] Tests de requêtes RAG (smoke test multilingue)...")

    queries = [
        ("EN", "My account is locked and I cannot log in"),
        ("FR", "Mon abonnement a été débité deux fois ce mois"),
        ("DE", "Das Produkt funktioniert nicht nach dem Update"),
        ("ES", "No puedo restablecer mi contraseña"),
        ("Général", "billing problem invoice not received"),
    ]

    for lang, q in queries:
        print(f"\n  [{lang}] {q}")
        try:
            res = rag.query(q)
            answer = res.get("answer", "").strip()
            sources = res.get("sources", [])
            # Tronquer la réponse pour l'affichage
            short_answer = answer[:300] + ("..." if len(answer) > 300 else "")
            print(f"  → Réponse ({len(answer)} chars): {short_answer}")
            print(f"  → Sources : {len(sources)} chunks | top sim = {sources[0]['similarity']:.1%}" if sources else "  → Aucune source")
        except Exception as e:
            print(f"  ✗ Erreur : {e}")


def run_evaluation(rag: "RAGEngine", sample_csv: Path, n_samples: int = 10) -> dict:
    """Évalue le pipeline (n_samples questions) et retourne le rapport."""
    from evaluate import RAGEvaluator

    print(f"\n[5/5] Évaluation ({n_samples} questions, LLM-as-judge)...")

    evaluator = RAGEvaluator(rag_engine=rag)

    # Générer les questions de test depuis l'échantillon
    test_cases = evaluator.generate_test_dataset(
        csv_path=sample_csv,
        n_samples=n_samples,
    )
    print(f"      → {len(test_cases)} questions générées")

    # Évaluation complète
    report = evaluator.run_full_evaluation(test_cases)

    # Sauvegarder le rapport
    out_dir = ROOT / "evaluation"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "multilingual_test_report.json"
    evaluator.save_report(report, str(report_path))

    return report


def print_report_summary(report: dict) -> float:
    """Affiche un résumé du rapport et retourne le score composite."""
    print("\n" + "=" * 62)
    print("  RAPPORT D'ÉVALUATION")
    print("=" * 62)

    summary = report.get("summary", report)

    # Métriques de retrieval
    retrieval = summary.get("retrieval_metrics", {})
    if retrieval:
        print("\n  Retrieval :")
        for k, v in retrieval.items():
            if isinstance(v, float):
                print(f"    {k:<25} {v:.3f}")

    # Métriques LLM-as-judge
    llm_metrics = summary.get("llm_metrics", {})
    if llm_metrics:
        print("\n  LLM-as-judge :")
        for k, v in llm_metrics.items():
            if isinstance(v, float):
                print(f"    {k:<25} {v:.3f}")

    # Score composite
    rag_score = summary.get("rag_score", summary.get("composite_score", None))
    if rag_score is None:
        # Calculer manuellement depuis les métriques disponibles
        scores = [v for v in {**retrieval, **llm_metrics}.values() if isinstance(v, float) and 0 <= v <= 1]
        rag_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n  ► RAG Score composite : {rag_score:.3f} / 1.000")

    threshold_pct = SATISFACTION_THRESHOLD * 100
    if rag_score >= SATISFACTION_THRESHOLD:
        print(f"  ✓ Score ≥ {threshold_pct:.0f}% → résultats SATISFAISANTS")
    else:
        print(f"  ✗ Score < {threshold_pct:.0f}% → résultats insuffisants")

    return rag_score


def main():
    parser = argparse.ArgumentParser(
        description="Test du pipeline RAG sur le dataset multilingual Kaggle"
    )
    parser.add_argument("--sample-size", type=int, default=2000,
                        help="Nombre de tickets à indexer (défaut : 2000)")
    parser.add_argument("--n-eval", type=int, default=10,
                        help="Nombre de questions d'évaluation (défaut : 10)")
    parser.add_argument("--gpt4o", action="store_true",
                        help="Utiliser GPT-4o directement (sans passer par le test mini)")
    parser.add_argument("--skip-download", type=str, default=None,
                        metavar="CSV_PATH",
                        help="Ignorer le téléchargement et utiliser ce CSV directement")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Ignorer l'étape d'évaluation LLM-as-judge")
    parser.add_argument("--skip-index", action="store_true",
                        help="Réutiliser le vectorstore existant sans ré-indexer")
    args = parser.parse_args()

    print("=" * 62)
    print("  RAG-TIME — TEST DATASET MULTILINGUAL")
    print("=" * 62)

    # ── Phase 1 : Téléchargement ─────────────────────────────────────────────
    if args.skip_download:
        sample_csv = Path(args.skip_download)
        df = pd.read_csv(sample_csv)
        df = _map_columns(df)
        df = _ensure_required_columns(df)
        df.to_csv(sample_csv, index=False)
        print(f"\n[1-2/5] CSV fourni directement : {sample_csv} ({len(df):,} lignes)")
    else:
        dataset_path = download_dataset()
        df, sample_csv = load_sample(dataset_path, sample_size=args.sample_size)

    # ── Phase 2 : Indexation ─────────────────────────────────────────────────
    use_gpt4o = args.gpt4o
    rag = build_index(sample_csv, use_gpt4o=use_gpt4o, skip_index=args.skip_index)

    # ── Phase 3 : Smoke tests ────────────────────────────────────────────────
    run_smoke_tests(rag)

    # ── Phase 4 : Évaluation ─────────────────────────────────────────────────
    if not args.skip_eval:
        try:
            report = run_evaluation(rag, sample_csv, n_samples=args.n_eval)
            rag_score = print_report_summary(report)
        except Exception as e:
            print(f"\n  ⚠ Évaluation échouée : {e}")
            print("     → Les smoke tests restent valides pour juger la qualité.")
            rag_score = None
    else:
        print("\n  (Évaluation LLM-as-judge ignorée)")
        rag_score = None

    # ── Phase 5 : Relance avec GPT-4o si satisfaisant ─────────────────────────
    if not use_gpt4o:
        if rag_score is not None:
            if rag_score >= SATISFACTION_THRESHOLD:
                print(f"\n{'=' * 62}")
                print("  RELANCE AVEC GPT-4o (score satisfaisant)")
                print(f"{'=' * 62}")
                rag_gpt4o = build_index(sample_csv, use_gpt4o=True, skip_index=args.skip_index)
                run_smoke_tests(rag_gpt4o)
                if not args.skip_eval:
                    try:
                        report_gpt4o = run_evaluation(rag_gpt4o, sample_csv, n_samples=args.n_eval)
                        print_report_summary(report_gpt4o)
                        # Sauvegarder aussi le rapport GPT-4o
                        gpt4o_path = ROOT / "evaluation" / "multilingual_gpt4o_report.json"
                        with open(gpt4o_path, "w", encoding="utf-8") as f:
                            json.dump(report_gpt4o, f, indent=2, ensure_ascii=False)
                        print(f"\n  Rapport GPT-4o sauvegardé : {gpt4o_path}")
                    except Exception as e:
                        print(f"  ⚠ Évaluation GPT-4o échouée : {e}")
            else:
                print(f"\n  → Score trop bas ({rag_score:.3f}), GPT-4o non lancé.")
                print("     Pistes d'amélioration : augmenter --sample-size, vérifier le mapping des colonnes.")
        else:
            # Pas d'éval → on demande à l'utilisateur s'il veut relancer en GPT-4o
            print("\n  → Évaluation non disponible.")
            print("     Pour lancer directement en GPT-4o : python test_multilingual_pipeline.py --gpt4o")

    print("\n✓ Pipeline terminé.")


if __name__ == "__main__":
    main()
