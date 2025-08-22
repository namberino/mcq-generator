from dataclasses import dataclass

"""MCQ Validation Scorer - workflow and usage

Purpose
-------
This module implements a compact, configurable workflow to convert raw
validation outputs (produced by `RAGMCQ.validate_mcqs`) into actionable
quality decisions and batch-level metrics. It centralizes scoring logic,
thresholds, and recommendations so callers can consistently decide whether
to approve, review, or reject generated MCQs.

Quick workflow
--------------
1. Use `RAGMCQ.generate_from_pdf(...)` or `RAGMCQ.generate_from_qdrant(...)`
	to create MCQs, then run `RAGMCQ.validate_mcqs(...)` to produce per-question
	validation data.
2. Instantiate `MCQValidationScorer` (optionally with a custom
	`ValidationConfig`) and call `process_batch_validation(...)` with a dict
	containing two keys: `'mcqs'` (the original questions) and
	`'validation'` (the report returned by `RAGMCQ.validate_mcqs`).
3. `process_batch_validation` will compute per-question scores, produce a
	decision (`APPROVE`, `REVIEW_REQUIRED`, etc.), aggregate batch metrics,
	and return recommendations.

Data contract / expected shapes
-------------------------------
- mcqs: Dict[str, Any]
    - mapping of question id (string) -> question dict (must contain
        localized keys like 'câu hỏi', 'lựa chọn', 'đáp án').
- validation report (per question): Dict with at least the keys
    - `max_similarity`: float (0.0-1.0)
    - `supported_by_embeddings`: bool
    - `evidence`: List[Dict] where each dict contains `idx`, `page`, `score`, `text`
    - `model_verdict`: dict or error object. Expected fields when valid: `supported` (bool),
        `confidence` (0.0-1.0), `evidence` (str), `reason` (str)

Configuration
-------------
All scoring and thresholds are centralized in `ValidationConfig`:
- `embedding_weight`, `model_weight`, `evidence_weight` (must sum to 1.0)
- similarity and evidence cutoffs
- category thresholds (excellent/good/acceptable/questionable)

Primary public API
------------------
- `MCQValidationScorer.calculate_validation_score(validation_data)` -> float (0-100)
- `MCQValidationScorer.make_quality_decision(validation_data)` -> dict (decision payload)
- `MCQValidationScorer.process_batch_validation({'mcqs':..., 'validation':...})` -> batch report
- `generate_quick_summary(processed_validation)` -> compact summary for API responses

Edge cases & notes
------------------
- If `model_verdict` is missing or contains an `error` key, the model component
    score defaults to 0 for that question.
- If no evidence chunks are present, the evidence component is 0.
- Be mindful of prompt/token limits: the verifier's `context_text` originates
    from the top-k retrieved chunks (see `RAGMCQ.validate_mcqs`) and can be long.
- Adjust `ValidationConfig` weights and thresholds to match your risk tolerance
    (e.g., raise `excellent_threshold` for stricter acceptance).

Example usage
-------------
>>> rag = RAGMCQ(...)
>>> mcqs = rag.generate_from_pdf('document.pdf', n_questions=10)
>>> validation = rag.validate_mcqs(mcqs)
>>> scorer = MCQValidationScorer()
>>> processed = scorer.process_batch_validation({'mcqs': mcqs, 'validation': validation})
>>> summary = generate_quick_summary(processed)

This module focuses on deterministic, explainable scoring and decision rules.
Tune `ValidationConfig` and, if necessary, extend scoring helpers to fit your
domain requirements.
"""


@dataclass
class ValidationConfig:
    """
    Configuration class for MCQ validation system
    """
    # Scoring weights (must sum to 1.0)
    embedding_weight: float = 0.4
    model_weight: float = 0.5
    evidence_weight: float = 0.1

    # Similarity thresholds
    similarity_threshold: float = 0.5
    evidence_cutoff: float = 0.5

    # Score thresholds for categories
    excellent_threshold: float = 85.0
    good_threshold: float = 70.0
    acceptable_threshold: float = 55.0
    questionable_threshold: float = 40.0

    require_model_verification: bool = True

    # Filtering defaults
    default_pass_rate: float = 0.7

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()

    def validate(self):
        """Validate configuration parameters"""
        # Check weights sum to 1.0
        total_weight = self.embedding_weight + self.model_weight + self.evidence_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Check thresholds are in correct order
        thresholds = [
            self.questionable_threshold,
            self.acceptable_threshold,
            self.good_threshold,
            self.excellent_threshold
        ]

        if thresholds != sorted(thresholds):
            raise ValueError("Thresholds must be in ascending order")

        # Check ranges
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if not (0.0 <= self.evidence_cutoff <= 1.0):
            raise ValueError("evidence_cutoff must be between 0.0 and 1.0")


class MCQValidationScorer:
    """
    Comprehensive MCQ validation scoring system combining embedding similarity
    and LLM model verification with configurable weights and thresholds.
    """

    def __init__(self, config=None):
        if config is None:
            self.config = ValidationConfig()
        elif isinstance(config, dict):
            # Convert dict config to ValidationConfig
            self.config = ValidationConfig(**config)
        else:
            self.config = config

    def calculate_validation_score(self, validation_data):
        """
        Calculate comprehensive validation score (0-100)

        Args:
            validation_data: Dict containing validation results from RAGMCQ.validate_mcqs()

        Returns:
            float: Validation score between 0-100
        """
        # Extract components
        max_similarity = validation_data.get("max_similarity", 0.0)
        supported_by_embeddings = validation_data.get("supported_by_embeddings", False)
        evidence = validation_data.get("evidence", [])
        model_verdict = validation_data.get("model_verdict", {})

        # Calculate component scores
        embedding_score = self._calculate_embedding_score(max_similarity, supported_by_embeddings)
        model_score = self._calculate_model_score(model_verdict)
        evidence_score = self._calculate_evidence_score(evidence)

        # Weighted total
        total_score = (
            embedding_score * self.config.embedding_weight * 100 +
            model_score * self.config.model_weight * 100 +
            evidence_score * self.config.evidence_weight * 100
        )

        return min(100.0, max(0.0, total_score))

    def _calculate_embedding_score(self, max_similarity, supported_by_embeddings):
        """Calculate embedding component score (0.0-1.0)"""
        base_score = max_similarity * 0.875  # Scale to 87.5% of component
        support_bonus = 0.125 if supported_by_embeddings else 0  # 12.5% bonus
        return base_score + support_bonus

    def _calculate_model_score(self, model_verdict):
        """Calculate model verification component score (0.0-1.0)"""
        if not model_verdict or "error" in model_verdict:
            return 0.0

        supported = model_verdict.get("supported", False)
        confidence = model_verdict.get("confidence", 0.0)

        if not supported:
            return confidence * 0.3  # Unsupported answers get max 30% of component

        # Supported answers: 50% base + 50% confidence-based
        return 0.5 + (confidence * 0.5)

    def _calculate_evidence_score(self, evidence):
        """Calculate evidence quality component score (0.0-1.0)"""
        if not evidence:
            return 0.0

        # Quantity component (up to 50% of evidence score)
        num_evidence = len(evidence)
        quantity_score = min(0.5, num_evidence * 0.125)  # Max 4 pieces for full quantity score

        # Quality component (up to 50% of evidence score)
        avg_score = sum(e.get("score", 0.0) for e in evidence) / len(evidence)
        quality_score = avg_score * 0.5

        return quantity_score + quality_score

    def get_validation_category(self, score):
        """
        Categorize validation score into quality levels

        Returns:
            tuple: (category, description, color_code)
        """
        if score >= self.config.excellent_threshold:
            return "EXCELLENT", "High confidence - Ready for use", "green"
        elif score >= self.config.good_threshold:
            return "GOOD", "Medium-high confidence - Minor review recommended", "lightgreen"
        elif score >= self.config.acceptable_threshold:
            return "ACCEPTABLE", "Medium confidence - Review recommended", "yellow"
        elif score >= self.config.questionable_threshold:
            return "QUESTIONABLE", "Low confidence - Significant review needed", "orange"
        else:
            return "POOR", "Very low confidence - Consider regenerating", "red"

    def make_quality_decision(self, validation_data):
        """
        Make actionable quality decision based on validation data

        Returns:
            Dict with decision, action, reasoning, and priority
        """
        score = self.calculate_validation_score(validation_data)
        category, description, color = self.get_validation_category(score)

        model_verdict = validation_data.get("model_verdict", {})
        max_similarity = validation_data.get("max_similarity", 0.0)
        evidence_count = len(validation_data.get("evidence", []))

        # Decision logic
        if score >= self.config.excellent_threshold:
            return {
                "decision": "APPROVE",
                "action": "Use as-is",
                "reasoning": f"Excellent validation score ({score:.1f}). High confidence in accuracy.",
                "priority": "LOW",
                "score": score,
                "category": category
            }

        elif score >= self.config.good_threshold:
            return {
                "decision": "APPROVE_WITH_REVIEW",
                "action": "Minor review recommended",
                "reasoning": f"Good validation score ({score:.1f}). Consider quick review for optimization.",
                "priority": "LOW",
                "score": score,
                "category": category
            }

        elif score >= self.config.acceptable_threshold:
            if model_verdict.get("supported", False) and model_verdict.get("confidence", 0) >= 0.8:
                return {
                    "decision": "CONDITIONAL_APPROVE",
                    "action": "Review model evidence",
                    "reasoning": f"Acceptable score ({score:.1f}) but strong model support. Review context alignment.",
                    "priority": "MEDIUM",
                    "score": score,
                    "category": category
                }
            else:
                return {
                    "decision": "REVIEW_REQUIRED",
                    "action": "Manual review needed",
                    "reasoning": f"Acceptable score ({score:.1f}) but weak model support. Manual verification needed.",
                    "priority": "HIGH",
                    "score": score,
                    "category": category
                }

        elif score >= self.config.questionable_threshold:
            return {
                "decision": "REJECT_WITH_FEEDBACK",
                "action": "Regenerate with feedback",
                "reasoning": f"Low score ({score:.1f}). Evidence: {evidence_count} chunks, similarity: {max_similarity:.2f}",
                "priority": "HIGH",
                "score": score,
                "category": category
            }

        else:
            return {
                "decision": "REJECT",
                "action": "Regenerate completely",
                "reasoning": f"Very low score ({score:.1f}). Poor evidence support and model confidence.",
                "priority": "CRITICAL",
                "score": score,
                "category": category
            }

    def process_batch_validation(self, mcqs_with_validation):
        """
        Process complete batch of MCQs with validation decisions

        Args:
            mcqs_with_validation: Dict with 'mcqs' and 'validation' keys

        Returns:
            Comprehensive validation report
        """
        results = {
            "processed_questions": {},
            "batch_summary": {
                "total": 0,
                "approved": 0,
                "conditional": 0,
                "review_required": 0,
                "rejected": 0,
                "average_score": 0.0,
                "score_distribution": {}
            },
            "recommendations": [],
            "quality_metrics": {
                "pass_rate": 0.0,
                "high_quality_rate": 0.0,
                "needs_attention": []
            }
        }

        scores = []
        category_counts = {"EXCELLENT": 0, "GOOD": 0, "ACCEPTABLE": 0, "QUESTIONABLE": 0, "POOR": 0}

        # Process each question
        for qid, validation_data in mcqs_with_validation["validation"].items():
            if qid not in mcqs_with_validation["mcqs"]:
                continue

            score = self.calculate_validation_score(validation_data)
            decision_data = self.make_quality_decision(validation_data)
            category = decision_data["category"]

            scores.append(score)
            category_counts[category] += 1

            results["processed_questions"][qid] = {
                "mcq": mcqs_with_validation["mcqs"][qid],
                "validation_score": score,
                "decision": decision_data,
                "validation_details": validation_data
            }

            # Update batch summary
            results["batch_summary"]["total"] += 1
            decision_type = decision_data["decision"].lower()

            if "approve" in decision_type and "conditional" not in decision_type:
                results["batch_summary"]["approved"] += 1
            elif "conditional" in decision_type:
                results["batch_summary"]["conditional"] += 1
            elif "review" in decision_type:
                results["batch_summary"]["review_required"] += 1
            else:
                results["batch_summary"]["rejected"] += 1

        # Calculate metrics
        if scores:
            results["batch_summary"]["average_score"] = sum(scores) / len(scores)
            results["batch_summary"]["score_distribution"] = category_counts

            total = results["batch_summary"]["total"]
            passed = results["batch_summary"]["approved"] + results["batch_summary"]["conditional"]
            high_quality = results["batch_summary"]["approved"]

            results["quality_metrics"]["pass_rate"] = passed / total if total > 0 else 0
            results["quality_metrics"]["high_quality_rate"] = high_quality / total if total > 0 else 0

            # Identify questions needing attention
            results["quality_metrics"]["needs_attention"] = [
                qid for qid, result in results["processed_questions"].items()
                if result["decision"]["priority"] in ["HIGH", "CRITICAL"]
            ]

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _generate_recommendations(self, results):
        """Generate actionable recommendations based on batch results"""
        recommendations = []

        metrics = results["quality_metrics"]
        summary = results["batch_summary"]

        if metrics["pass_rate"] < 0.5:
            recommendations.append({
                "type": "QUALITY_CONCERN",
                "severity": "HIGH",
                "message": f"Low pass rate ({metrics['pass_rate']:.1%}). Consider adjusting generation parameters.",
                "action": "Review generation settings, chunk quality, or source material"
            })

        if metrics["high_quality_rate"] > 0.8:
            recommendations.append({
                "type": "EXCELLENT_QUALITY",
                "severity": "INFO",
                "message": f"High quality rate ({metrics['high_quality_rate']:.1%}). Current settings are optimal.",
                "action": "Maintain current generation parameters"
            })

        if summary["rejected"] > summary["total"] * 0.3:
            recommendations.append({
                "type": "HIGH_REJECTION_RATE",
                "severity": "MEDIUM",
                "message": f"High rejection rate ({summary['rejected']}/{summary['total']}). Quality issues detected.",
                "action": "Review source material quality and consider stricter content filtering"
            })

        return recommendations


def generate_quick_summary(processed_validation):
    """
    Generate a quick validation summary for API responses

    Returns:
        Dict with essential validation metrics
    """
    summary = processed_validation["batch_summary"]

    return {
        "total_questions": summary["total"],
        "validation_summary": {
            "passed": summary["approved"] + summary["conditional"],
            "failed": summary["rejected"] + summary["review_required"],
            "pass_rate": f"{((summary['approved'] + summary['conditional']) / summary['total']):.1%}" if summary["total"] > 0 else "0%",
            "average_score": f"{summary['average_score']:.1f}"
        },
        "quality_distribution": {
            "excellent": summary["score_distribution"].get("EXCELLENT", 0),
            "good": summary["score_distribution"].get("GOOD", 0),
            "acceptable": summary["score_distribution"].get("ACCEPTABLE", 0),
            "questionable": summary["score_distribution"].get("QUESTIONABLE", 0),
            "poor": summary["score_distribution"].get("POOR", 0)
        },
        "top_recommendations": processed_validation["recommendations"][:3],  # Top 3 recommendations
        "needs_attention": processed_validation["quality_metrics"]["needs_attention"]
    }
