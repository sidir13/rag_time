"""Évaluation complète du pipeline RAG — Métriques de A à Z
===========================================================

Métriques implémentées :

  1. Retrieval (sans LLM) :
     - Hit Rate @ K     : le ticket de référence est-il dans le top-K ?
     - MRR @ K          : Mean Reciprocal Rank
     - Precision @ K    : chunks pertinents récupérés / K
     - NDCG @ K         : Normalized Discounted Cumulative Gain
     - Mean Similarity  : score cosinus moyen des résultats récupérés

  2. LLM-as-judge (via OpenRouter) :
     - Faithfulness        : l'answer est-elle fidèle au contexte ?
     - Answer Relevance    : l'answer répond-elle à la question ?
     - Context Precision   : les chunks récupérés sont-ils pertinents ?
     - Context Recall      : le contexte couvre-t-il la réponse attendue ?
     - Hallucination Score : l'answer contient-elle des informations inventées ?
     - Completeness        : l'answer est-elle complète ?

  3. Score composite :
     - RAG Score : moyenne pondérée des métriques LLM (excl. hallucination)

Usage :
    from src.evaluate import RAGEvaluator
    evaluator = RAGEvaluator(rag_engine)
    test_cases = evaluator.generate_test_dataset(csv_path, n_samples=50)
    report = evaluator.run_full_evaluation(test_cases)
    evaluator.save_report(report, "evaluation/report.json")
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class RAGEvaluator:
    """Évaluateur complet du pipeline RAG.

    Paramètres :
        rag_engine   : instance de RAGEngine (src/rag.py)
        judge_model  : modèle LLM pour les métriques LLM-as-judge
        rate_limit_delay : délai (secondes) entre les appels LLM (évite 429)
    """

    def __init__(
        self,
        rag_engine,
        judge_model: str = JUDGE_MODEL,
        rate_limit_delay: float = 0.5,
    ):
        self.rag = rag_engine
        self.judge_model = judge_model
        self.rate_limit_delay = rate_limit_delay
        self._llm: Optional[OpenAI] = None

        api_key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
        if api_key:
            self._llm = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

    # ── Génération du dataset de test ───────────────────────────────────────

    def generate_test_dataset(
        self,
        csv_path: str | Path,
        n_samples: int = 50,
        random_state: int = 42,
        indexed_ids: set[str] | None = None,
    ) -> list[dict]:
        """Génère automatiquement des paires (question, ticket_id) via LLM.

        Pour chaque ticket samplé, le LLM crée une question réaliste
        qu'un agent de support poserait pour trouver ce ticket.

        Args:
            csv_path     : chemin du CSV source
            n_samples    : nombre de tickets à échantillonner
            random_state : graine aléatoire
            indexed_ids  : si fourni, on échantillonne UNIQUEMENT parmi ces IDs
                           (évite de tester des tickets absents de l'index)

        Returns:
            Liste de dicts : {ticket_id, question, ground_truth_issue,
                              ground_truth_resolution, metadata}
        """
        if not self._llm:
            raise ValueError(
                "OPENROUTER_API_KEY non configurée — LLM requis pour générer les questions"
            )

        df = pd.read_csv(csv_path)
        df["ticket_id"] = df["ticket_id"].astype(str)

        # Filtrer aux seuls tickets présents dans l'index
        if indexed_ids:
            df_pool = df[df["ticket_id"].isin(indexed_ids)]
            if df_pool.empty:
                raise ValueError(
                    "Aucun ticket du CSV ne correspond aux IDs de l'index. "
                    "Vérifiez que l'index a été construit à partir du même CSV."
                )
            print(f"  Pool de tickets indexés : {len(df_pool):,} / {len(df):,}")
        else:
            df_pool = df
            print(
                "  ATTENTION : indexed_ids non fourni — on sample sur tout le CSV.\n"
                "  Les métriques de retrieval risquent d'être toutes à 0."
            )

        sample = df_pool.sample(n=min(n_samples, len(df_pool)), random_state=random_state)

        test_cases = []
        for _, row in sample.iterrows():
            ticket_id = str(row.get("ticket_id", ""))
            issue = str(row.get("issue_description", ""))
            resolution = str(row.get("resolution_notes", ""))
            product = str(row.get("product", ""))
            category = str(row.get("category", ""))

            question = self._generate_question(issue, resolution, product, category)
            if question:
                test_cases.append(
                    {
                        "ticket_id": ticket_id,
                        "question": question,
                        "ground_truth_issue": issue,
                        "ground_truth_resolution": resolution,
                        "metadata": {
                            "product": product,
                            "category": category,
                            "priority": str(row.get("priority", "")),
                            "status": str(row.get("status", "")),
                        },
                    }
                )
            time.sleep(self.rate_limit_delay)

        return test_cases

    def _generate_question(
        self, issue: str, resolution: str, product: str, category: str
    ) -> str | None:
        """Génère une question réaliste pour un ticket donné."""
        prompt = (
            "You are a customer support agent. Based on this support ticket, "
            "generate ONE realistic question that a support agent might search for "
            "to find similar past tickets.\n\n"
            f"Product: {product}\nCategory: {category}\n"
            f"Issue: {issue[:300]}\nResolution: {resolution[:200]}\n\n"
            "Generate ONLY the question (1 sentence, in English), nothing else."
        )
        try:
            resp = self._llm.chat.completions.create(
                model=self.judge_model,
                max_tokens=120,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            # Fallback : paraphrase de l'issue
            return f"How to resolve: {issue[:150]}?" if issue else None

    # ── Métriques de Retrieval ───────────────────────────────────────────────

    def evaluate_retrieval(
        self,
        test_cases: list[dict],
        k_values: list[int] | None = None,
    ) -> dict:
        """Calcule les métriques de retrieval pour un ensemble de test.

        Métriques :
        - hit_rate@K    : fraction de questions où le ticket de référence est dans le top-K
        - mrr@K         : Mean Reciprocal Rank
        - precision@K   : fraction des top-K chunks issus du ticket de référence
        - ndcg@K        : Normalized Discounted Cumulative Gain (binary relevance)
        - mean_sim@5    : similarité cosinus moyenne des 5 premiers résultats
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        buckets: dict[int, dict] = {
            k: {"hits": 0, "reciprocal_ranks": [], "ndcg": [], "precision": []}
            for k in k_values
        }
        sim_scores: list[float] = []
        max_k = max(k_values)

        for case in test_cases:
            ticket_id = case["ticket_id"]
            retrieved = self.rag.retrieve(case["question"], top_k=max_k)

            # Normalise les IDs : "{ticket_id}__issue" → "ticket_id"
            base_ids = [
                m.get("ticket_id", "").split("__")[0]
                for m in retrieved["metadatas"]
            ]

            if retrieved["similarities"]:
                sim_scores.append(float(np.mean(retrieved["similarities"][:5])))

            # Rang du ticket de référence (1-indexé, 0 si absent)
            try:
                rank = base_ids.index(ticket_id) + 1
            except ValueError:
                rank = 0

            for k in k_values:
                top_k_ids = base_ids[:k]
                buckets[k]["hits"] += int(ticket_id in top_k_ids)

                # MRR
                buckets[k]["reciprocal_ranks"].append(
                    1.0 / rank if 0 < rank <= k else 0.0
                )

                # Precision@K
                n_relevant = sum(1 for tid in top_k_ids if tid == ticket_id)
                buckets[k]["precision"].append(n_relevant / k)

                # NDCG@K (relevance binaire)
                gains = [1.0 if tid == ticket_id else 0.0 for tid in top_k_ids]
                dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
                ideal_dcg = 1.0 / math.log2(2)  # best possible: hit at rank 1
                buckets[k]["ndcg"].append(min(dcg / ideal_dcg, 1.0))

        n = len(test_cases)
        agg: dict = {}
        for k in k_values:
            agg[f"hit_rate@{k}"] = buckets[k]["hits"] / n
            agg[f"mrr@{k}"] = float(np.mean(buckets[k]["reciprocal_ranks"]))
            agg[f"precision@{k}"] = float(np.mean(buckets[k]["precision"]))
            agg[f"ndcg@{k}"] = float(np.mean(buckets[k]["ndcg"]))

        agg["mean_similarity@5"] = float(np.mean(sim_scores)) if sim_scores else 0.0
        return agg

    # ── LLM-as-judge : helpers ───────────────────────────────────────────────

    def _judge(self, prompt: str, max_tokens: int = 250) -> dict:
        """Appelle le LLM juge et parse la réponse JSON."""
        if not self._llm:
            return {"score": None, "explanation": "LLM non disponible"}

        try:
            resp = self._llm.chat.completions.create(
                model=self.judge_model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict evaluator. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content.strip()
            # Extract JSON block
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"score": None, "explanation": content}
        except Exception as exc:
            return {"score": None, "explanation": str(exc)}

    # ── LLM-as-judge : métriques ─────────────────────────────────────────────

    def evaluate_faithfulness(
        self, question: str, context: str, answer: str
    ) -> dict:
        """Fidélité : chaque affirmation de l'answer est-elle dans le context ?

        Score 0-1 :
          1.0 → toutes les affirmations sont supportées
          0.0 → l'answer contredit ou ignore le contexte
        """
        prompt = (
            "Evaluate the FAITHFULNESS of the answer given the context below.\n\n"
            f"CONTEXT:\n{context[:2500]}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:\n{answer[:1200]}\n\n"
            "Score 0-1 based on how well every claim in the answer is supported "
            "by the context (not by general knowledge).\n"
            "- 1.0 : all claims directly supported\n"
            "- 0.75: mostly supported, minor unsupported details\n"
            "- 0.5 : half supported\n"
            "- 0.25: mostly unsupported\n"
            "- 0.0 : answer contradicts or ignores the context\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "explanation": "<1 sentence>"}'
        )
        result = self._judge(prompt)
        time.sleep(self.rate_limit_delay)
        return {
            "faithfulness": float(result["score"]) if result.get("score") is not None else None,
            "faithfulness_reason": result.get("explanation", ""),
        }

    def evaluate_answer_relevance(self, question: str, answer: str) -> dict:
        """Pertinence : l'answer répond-elle directement à la question ?

        Score 0-1 : 1.0 = réponse parfaitement pertinente.
        """
        prompt = (
            "Evaluate the RELEVANCE of the answer to the question.\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:\n{answer[:1200]}\n\n"
            "Score 0-1:\n"
            "- 1.0 : directly and completely answers the question\n"
            "- 0.75: mostly answers with minor gaps\n"
            "- 0.5 : partially answers\n"
            "- 0.25: tangentially related\n"
            "- 0.0 : completely irrelevant\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "explanation": "<1 sentence>"}'
        )
        result = self._judge(prompt)
        time.sleep(self.rate_limit_delay)
        return {
            "answer_relevance": float(result["score"]) if result.get("score") is not None else None,
            "answer_relevance_reason": result.get("explanation", ""),
        }

    def evaluate_context_precision(
        self, question: str, chunks: list[str]
    ) -> dict:
        """Précision du contexte : quelle fraction des top-5 chunks est pertinente ?

        Score 0-1 : 1.0 = tous les chunks sont utiles pour répondre.
        """
        top5 = chunks[:5]
        if not top5:
            return {"context_precision": None}

        n_relevant = 0
        for chunk in top5:
            prompt = (
                "Is the following context chunk USEFUL for answering the question?\n\n"
                f"QUESTION: {question}\n\n"
                f"CHUNK:\n{chunk[:600]}\n\n"
                'Respond with JSON only: {"relevant": true or false, "reason": "<1 sentence>"}'
            )
            result = self._judge(prompt, max_tokens=100)
            time.sleep(self.rate_limit_delay)
            if result.get("relevant") is True:
                n_relevant += 1

        return {
            "context_precision": n_relevant / len(top5),
            "relevant_chunks": n_relevant,
            "total_chunks_evaluated": len(top5),
        }

    def evaluate_context_recall(
        self, question: str, context: str, ground_truth: str
    ) -> dict:
        """Recall du contexte : le contexte contient-il tout pour répondre ?

        Score 0-1 : 1.0 = le contexte permet de reproduire la réponse attendue.
        """
        prompt = (
            "Evaluate if the CONTEXT contains all the information needed to produce "
            "the expected answer.\n\n"
            f"QUESTION: {question}\n\n"
            f"EXPECTED ANSWER (ground truth):\n{ground_truth[:600]}\n\n"
            f"RETRIEVED CONTEXT:\n{context[:2500]}\n\n"
            "Score 0-1:\n"
            "- 1.0 : context fully covers the expected answer\n"
            "- 0.75: mostly covers it\n"
            "- 0.5 : partial coverage\n"
            "- 0.25: little coverage\n"
            "- 0.0 : completely missing\n\n"
            'Respond with JSON only: {"score": <float 0-1>, "explanation": "<1 sentence>"}'
        )
        result = self._judge(prompt)
        time.sleep(self.rate_limit_delay)
        return {
            "context_recall": float(result["score"]) if result.get("score") is not None else None,
            "context_recall_reason": result.get("explanation", ""),
        }

    def evaluate_hallucination(self, context: str, answer: str) -> dict:
        """Hallucination : l'answer invente-t-elle des informations absentes du contexte ?

        Score 0-1 : 0.0 = aucune hallucination, 1.0 = hallucination sévère.
        """
        prompt = (
            "Detect HALLUCINATIONS in the answer — information not present in the context.\n\n"
            f"CONTEXT:\n{context[:2500]}\n\n"
            f"ANSWER:\n{answer[:1200]}\n\n"
            "Score 0-1 for hallucination level:\n"
            "- 0.0 : no hallucinations\n"
            "- 0.25: minor unsupported details\n"
            "- 0.5 : several invented facts\n"
            "- 0.75: most content is invented\n"
            "- 1.0 : entirely fabricated\n\n"
            'Respond with JSON only: {"score": <float 0-1>, '
            '"hallucinated_claims": ["<claim>", ...], "explanation": "<1 sentence>"}'
        )
        result = self._judge(prompt)
        time.sleep(self.rate_limit_delay)
        return {
            "hallucination_score": float(result["score"]) if result.get("score") is not None else None,
            "hallucinated_claims": result.get("hallucinated_claims", []),
            "hallucination_reason": result.get("explanation", ""),
        }

    def evaluate_completeness(self, question: str, answer: str) -> dict:
        """Complétude : l'answer traite-t-elle tous les aspects de la question ?

        Score 0-1 : 1.0 = réponse exhaustive.
        """
        prompt = (
            "Evaluate the COMPLETENESS of the answer — does it address all aspects "
            "of the question?\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:\n{answer[:1200]}\n\n"
            "Score 0-1:\n"
            "- 1.0 : all aspects addressed thoroughly\n"
            "- 0.75: most aspects addressed\n"
            "- 0.5 : about half addressed\n"
            "- 0.25: barely addresses the question\n"
            "- 0.0 : completely incomplete\n\n"
            'Respond with JSON only: {"score": <float 0-1>, '
            '"missing_aspects": ["<aspect>", ...], "explanation": "<1 sentence>"}'
        )
        result = self._judge(prompt)
        time.sleep(self.rate_limit_delay)
        return {
            "completeness": float(result["score"]) if result.get("score") is not None else None,
            "missing_aspects": result.get("missing_aspects", []),
            "completeness_reason": result.get("explanation", ""),
        }

    # ── Pipeline par question ────────────────────────────────────────────────

    def evaluate_single(self, test_case: dict, top_k: int = 5) -> dict:
        """Évalue un seul test case avec toutes les métriques disponibles.

        Returns :
            Dict avec métriques de retrieval + métriques LLM (si disponible)
        """
        question = test_case["question"]
        ticket_id = test_case["ticket_id"]
        ground_truth = test_case.get("ground_truth_resolution", "")

        # ── Retrieval ────────────────────────────────────────────────────────
        retrieved = self.rag.retrieve(question, top_k=top_k)
        context = self.rag.build_context(retrieved)

        base_ids = [
            m.get("ticket_id", "").split("__")[0]
            for m in retrieved["metadatas"]
        ]

        try:
            rank = base_ids.index(ticket_id) + 1
        except ValueError:
            rank = 0

        retrieval = {
            "hit": rank > 0,
            "rank": rank,
            "reciprocal_rank": 1.0 / rank if rank > 0 else 0.0,
            "mean_similarity": float(np.mean(retrieved["similarities"])) if retrieved["similarities"] else 0.0,
            "top1_similarity": float(retrieved["similarities"][0]) if retrieved["similarities"] else 0.0,
        }

        # ── Génération ───────────────────────────────────────────────────────
        answer = ""
        if self.rag.has_llm:
            answer = self.rag._generate(context, question, temperature=0.0)

        # ── Métriques LLM ────────────────────────────────────────────────────
        llm_metrics: dict = {}
        if self._llm and answer:
            llm_metrics.update(self.evaluate_faithfulness(question, context, answer))
            llm_metrics.update(self.evaluate_answer_relevance(question, answer))
            llm_metrics.update(self.evaluate_context_precision(question, retrieved["documents"]))
            if ground_truth:
                llm_metrics.update(self.evaluate_context_recall(question, context, ground_truth))
            llm_metrics.update(self.evaluate_hallucination(context, answer))
            llm_metrics.update(self.evaluate_completeness(question, answer))

            # Score composite : moyenne des bonnes métriques (hors hallucination inversée)
            scores = [
                llm_metrics.get("faithfulness"),
                llm_metrics.get("answer_relevance"),
                llm_metrics.get("context_precision"),
                llm_metrics.get("context_recall"),
                (1.0 - llm_metrics["hallucination_score"])
                if llm_metrics.get("hallucination_score") is not None
                else None,
                llm_metrics.get("completeness"),
            ]
            valid = [s for s in scores if s is not None]
            llm_metrics["rag_score"] = float(np.mean(valid)) if valid else None

        return {
            "ticket_id": ticket_id,
            "question": question,
            "answer_excerpt": answer[:400] + "…" if len(answer) > 400 else answer,
            **retrieval,
            **llm_metrics,
        }

    # ── Pipeline d'évaluation complet ────────────────────────────────────────

    def run_full_evaluation(
        self,
        test_cases: list[dict],
        top_k: int = 5,
        verbose: bool = True,
    ) -> dict:
        """Lance l'évaluation complète sur tous les test cases.

        Returns :
            Dict avec summary (métriques agrégées) + individual_results
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(test_cases, desc="Évaluation RAG")
        except ImportError:
            iterator = test_cases

        individual: list[dict] = []
        for case in iterator:
            result = self.evaluate_single(case, top_k=top_k)
            individual.append(result)
            if verbose:
                hit_str = f"rank={result['rank']}" if result["hit"] else "MISS"
                rag = result.get("rag_score")
                rag_str = f"{rag:.2f}" if isinstance(rag, float) else "N/A"
                print(
                    f"  [{hit_str}] RAG={rag_str} | {case['question'][:70]}..."
                )

        summary = self._aggregate(individual)

        report = {
            "summary": summary,
            "n_samples": len(individual),
            "top_k": top_k,
            "judge_model": self.judge_model,
            "individual_results": individual,
        }

        if verbose:
            self._print_summary(summary)

        return report

    def _aggregate(self, results: list[dict]) -> dict:
        """Agrège les métriques individuelles (moyenne, ignorant les None)."""

        def mean_of(key: str) -> float | None:
            vals = [r[key] for r in results if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        metrics = [
            "hit", "reciprocal_rank", "mean_similarity", "top1_similarity",
            "faithfulness", "answer_relevance",
            "context_precision", "context_recall",
            "hallucination_score", "completeness", "rag_score",
        ]
        return {k: mean_of(k) for k in metrics if mean_of(k) is not None}

    def _print_summary(self, s: dict) -> None:
        print("\n" + "=" * 62)
        print("  RAG EVALUATION REPORT")
        print("=" * 62)
        print("\n  Retrieval Metrics :")
        print(f"    Hit Rate            : {s.get('hit', 'N/A'):.1%}")
        print(f"    MRR                 : {s.get('reciprocal_rank', 'N/A'):.4f}")
        print(f"    Mean Similarity @5  : {s.get('mean_similarity', 'N/A'):.4f}")
        print(f"    Top-1 Similarity    : {s.get('top1_similarity', 'N/A'):.4f}")
        if "faithfulness" in s:
            print("\n  LLM-as-Judge Metrics :")
            print(f"    Faithfulness        : {s.get('faithfulness', 'N/A'):.2f} / 1.0")
            print(f"    Answer Relevance    : {s.get('answer_relevance', 'N/A'):.2f} / 1.0")
            print(f"    Context Precision   : {s.get('context_precision', 'N/A'):.2f} / 1.0")
            if "context_recall" in s:
                print(f"    Context Recall      : {s.get('context_recall', 'N/A'):.2f} / 1.0")
            print(f"    Hallucination       : {s.get('hallucination_score', 'N/A'):.2f}  (↓ lower=better)")
            print(f"    Completeness        : {s.get('completeness', 'N/A'):.2f} / 1.0")
            print(f"\n  Overall RAG Score   : {s.get('rag_score', 'N/A'):.2f} / 1.0")
        print("=" * 62 + "\n")

    # ── Sauvegarde ───────────────────────────────────────────────────────────

    def save_report(
        self,
        report: dict,
        output_path: str | Path = "evaluation/report.json",
    ) -> None:
        """Sauvegarde le rapport en JSON (détaillé) et CSV (résumé par question)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON complet
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # CSV résumé par question
        csv_path = output_path.with_suffix(".csv")
        rows = []
        keep_cols = [
            "ticket_id", "question", "hit", "rank", "reciprocal_rank",
            "mean_similarity", "top1_similarity",
            "faithfulness", "answer_relevance",
            "context_precision", "context_recall",
            "hallucination_score", "completeness", "rag_score",
        ]
        for r in report.get("individual_results", []):
            rows.append({col: r.get(col) for col in keep_cols})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        print(f"\n  Rapport complet  : {output_path}")
        print(f"  Résumé CSV       : {csv_path}")
