"""Script principal d'évaluation du pipeline RAG
================================================

Étapes automatiques :
  1. Chargement du RAGEngine (index nomic-embed existant)
  2. Génération des questions de test via LLM (ou chargement depuis fichier)
  3. Évaluation de chaque question : retrieval + génération + LLM-as-judge
  4. Rapport JSON + CSV dans le dossier de sortie

Usage :
    python evaluate_rag.py
    python evaluate_rag.py --n-samples 100 --top-k 5
    python evaluate_rag.py --test-file evaluation/test_cases.json  # réutilise les questions
    python evaluate_rag.py --retrieval-only                         # skip la génération LLM

Options :
    --n-samples       : nombre de questions de test à générer (défaut : 50)
    --top-k           : chunks récupérés par requête (défaut : 5)
    --output          : dossier de sortie (défaut : evaluation/)
    --test-file       : chemin JSON vers des test cases pré-générés (évite d'appeler le LLM)
    --retrieval-only  : évalue uniquement les métriques de retrieval (pas de génération LLM)
    --judge-model     : modèle OpenRouter utilisé comme juge (défaut : openai/gpt-4o-mini)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ingest import Ingestor, DATA_RAW, VECTORSTORE_DIR  # noqa: E402
from rag import RAGEngine  # noqa: E402
from evaluate import RAGEvaluator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation complète du pipeline RAG")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Nombre de questions de test à générer (défaut : 50)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Nombre de chunks récupérés par requête (défaut : 5)")
    parser.add_argument("--output", type=str, default="evaluation",
                        help="Dossier de sortie pour les rapports (défaut : evaluation/)")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Fichier JSON de test cases pré-générés (skip la génération LLM)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Évalue uniquement les métriques de retrieval (sans LLM)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Modèle OpenRouter utilisé comme juge (ex: openai/gpt-4o-mini)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Chargement du RAGEngine ───────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RAG EVALUATION — nomic-ai/nomic-embed-text-v1.5")
    print("=" * 62)
    print("\n[1/4] Chargement du moteur RAG...")

    ingestor = Ingestor()
    if ingestor.count == 0:
        print(
            "\n  ERREUR : l'index est vide. Lancez d'abord :\n"
            "    python reindex.py\n"
        )
        sys.exit(1)

    rag_engine = RAGEngine(
        model=ingestor.model,
        index=ingestor.index,
        store=ingestor.store,
    )
    print(f"  → {rag_engine.index_count:,} chunks dans l'index")

    # IDs des tickets effectivement indexés
    # Pour le format 3-chunks, les IDs sont "{ticket_id}__issue" etc. → on strip le suffixe
    indexed_ids: set[str] = {
        mid.split("__")[0]
        for mid in rag_engine.store.get("ids", [])
    }
    print(f"  → {len(indexed_ids):,} tickets uniques dans l'index")

    if not rag_engine.has_llm:
        print(
            "\n  ATTENTION : OPENROUTER_API_KEY non configurée.\n"
            "  Les métriques LLM-as-judge ne seront pas disponibles.\n"
            "  Seules les métriques de retrieval seront calculées.\n"
        )

    # ── 2. Dataset de test ───────────────────────────────────────────────────
    kwargs = {}
    if args.judge_model:
        kwargs["judge_model"] = args.judge_model

    evaluator = RAGEvaluator(rag_engine, **kwargs)

    if args.test_file and Path(args.test_file).exists():
        print(f"\n[2/4] Chargement des test cases : {args.test_file}")
        with open(args.test_file, encoding="utf-8") as f:
            test_cases = json.load(f)
        print(f"  → {len(test_cases)} questions chargées")
    else:
        print(f"\n[2/4] Génération de {args.n_samples} questions de test via LLM...")
        if not rag_engine.has_llm and not args.retrieval_only:
            print("  Impossible de générer des questions sans OPENROUTER_API_KEY.")
            sys.exit(1)
        test_cases = evaluator.generate_test_dataset(
            csv_path=DATA_RAW,
            n_samples=args.n_samples,
            indexed_ids=indexed_ids,
        )
        # Sauvegarder pour réutilisation
        test_file_path = output_dir / "test_cases.json"
        with open(test_file_path, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        print(f"  → {len(test_cases)} questions sauvegardées dans {test_file_path}")

    # ── 3. Métriques de retrieval seules (rapide, sans LLM) ─────────────────
    print(f"\n[3/4] Calcul des métriques de retrieval (k=[1,3,5,10])...")
    retrieval_metrics = evaluator.evaluate_retrieval(test_cases, k_values=[1, 3, 5, 10])

    print("\n  Résultats retrieval :")
    for metric, value in retrieval_metrics.items():
        print(f"    {metric:<22} : {value:.4f}")

    # ── 4. Évaluation complète (retrieval + LLM-as-judge) ───────────────────
    if args.retrieval_only:
        # Sauvegarder uniquement les métriques de retrieval
        report = {
            "summary": retrieval_metrics,
            "n_samples": len(test_cases),
            "top_k": args.top_k,
            "mode": "retrieval_only",
            "individual_results": [],
        }
    else:
        print(f"\n[4/4] Évaluation complète (retrieval + LLM-as-judge, top_k={args.top_k})...")
        report = evaluator.run_full_evaluation(
            test_cases=test_cases,
            top_k=args.top_k,
            verbose=True,
        )
        # Merge retrieval-only metrics (all k values) into summary
        report["summary"].update(retrieval_metrics)

    # ── 5. Sauvegarde ────────────────────────────────────────────────────────
    output_path = output_dir / "evaluation_report.json"
    evaluator.save_report(report, output_path)

    print("\n  Évaluation terminée.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
