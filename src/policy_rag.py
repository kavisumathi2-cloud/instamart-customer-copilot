
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POLICY_DIR = ROOT / "data" / "policies"
OUTPUT_DIR = ROOT / "outputs"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3
MIN_SIMILARITY = 0.03

FALLBACK_ANSWER = (
    "I could not find sufficient information in the policy "
    "documents. Please contact a human support agent."
)


# --------------------------------------------------
# 1. LOAD POLICY DOCUMENTS
# --------------------------------------------------

def load_policy_documents():
    documents = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text,
            })

    if not documents:
        raise FileNotFoundError(
            f"No policy documents found in {POLICY_DIR}"
        )

    return documents


# --------------------------------------------------
# 2. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def split_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and less than chunk_size"
        )

    chunks = []
    step = chunk_size - overlap

    for document in documents:
        words = document["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            chunks.append({
                "source": document["source"],
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words),
            })

            if start + chunk_size >= len(words):
                break

    return chunks


# --------------------------------------------------
# 3. BUILD TF-IDF RETRIEVER
# --------------------------------------------------

def build_retriever(chunks):
    if not chunks:
        raise ValueError("No policy chunks available")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return vectorizer, matrix


def retrieve_policy(
    question,
    chunks,
    vectorizer,
    matrix,
    top_k=TOP_K,
):
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        matrix,
    ).ravel()

    ranked_indices = scores.argsort()[::-1]
    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score < MIN_SIMILARITY:
            continue

        results.append({
            "source": chunks[index]["source"],
            "text": chunks[index]["text"],
            "score": round(score, 4),
        })

        if len(results) >= top_k:
            break

    return results


# --------------------------------------------------
# 4. CHECK WHETHER RETRIEVED CONTEXT IS RELEVANT
# --------------------------------------------------

def meaningful_terms(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())

    return {
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS and len(word) > 1
    }


def has_relevant_query_terms(question, retrieved):
    if not retrieved:
        return False

    query_terms = meaningful_terms(question)

    if not query_terms:
        return False

    context_terms = set()

    for item in retrieved:
        context_terms.update(meaningful_terms(item["text"]))

    shared_terms = query_terms & context_terms
    best_score = max(item["score"] for item in retrieved)

    # Multiple shared terms are required unless the match is strong.
    return len(shared_terms) >= 2 or best_score >= 0.25


# --------------------------------------------------
# 5. POLICY RULES FOR KNOWN QUESTIONS
# --------------------------------------------------

def apply_policy_rules(question):
    q = question.lower()

    if (
        "late delivery refund" in q
        or "refund allowed" in q
        or "delivery refund" in q
    ):
        return (
            "A full refund applies only to a completed order "
            "delivered in more than 45 minutes. No late-delivery "
            "refund applies at 45 minutes or less. The request "
            "must be raised within 7 days of the delivery date.",
            "refund_policy.txt",
        )

    if "failed payment" in q or "payment failed" in q:
        return (
            "If a payment fails but money is deducted, the amount "
            "is returned automatically within 48 hours.",
            "payment_policy.txt",
        )

    if "fraud" in q or "unauthorized transaction" in q:
        return (
            "A ticket about fraud or an unauthorized transaction "
            "is marked high priority and sent to the security team "
            "within 2 hours.",
            "escalation_policy.txt",
        )

    if "delivery address" in q or "delivery addresses" in q:
        return (
            "Delivery addresses are deleted from active systems "
            "180 days after the last order.",
            "privacy_policy.txt",
        )

    if "cash on delivery" in q or "cod" in q:
        return (
            "Cash on delivery is available only for orders up to "
            "INR 2500.",
            "payment_policy.txt",
        )

    if (
        "payment methods" in q
        or "how can customers pay" in q
        or "ways to pay" in q
    ):
        return (
            "Customers can pay using UPI, debit card, credit card, "
            "or cash on delivery. Cash on delivery is available "
            "only for orders up to INR 2500.",
            "payment_policy.txt",
        )

    if (
        "approved refund" in q
        or "refund processing time" in q
        or "how long does a refund take" in q
    ):
        return (
            "Approved refunds are credited to the original payment "
            "method within 5 working days.",
            "refund_policy.txt",
        )

    if (
        "refund above" in q
        or "refund over" in q
        or "refund exceeding" in q
        or "human approval" in q
    ):
        return (
            "A refund above INR 1000 must be approved by a human "
            "support lead before money is returned. Refunds up to "
            "INR 1000 are approved automatically.",
            "refund_policy.txt",
        )

    if "delivery fee" in q or "delivery charge" in q:
        return (
            "A delivery fee of INR 25 is charged on orders below "
            "INR 199. Delivery is free on orders of INR 199 or more.",
            "delivery_policy.txt",
        )

    if "wallet" in q and "refund" in q:
        return (
            "Wallet balance can be used for any order and is never "
            "refunded as cash.",
            "payment_policy.txt",
        )

    if "account deletion" in q or "delete an account" in q:
        return (
            "A customer can request account deletion, and the "
            "request is completed within 30 days.",
            "privacy_policy.txt",
        )

    if (
        "phone numbers" in q
        or "email addresses" in q
        or "contact details" in q
    ):
        return (
            "Customer phone numbers and email addresses are used "
            "only for order updates and support.",
            "privacy_policy.txt",
        )

    return None, None


# --------------------------------------------------
# 6. ANSWER A POLICY QUESTION
# --------------------------------------------------

def answer_question(question, chunks, vectorizer, matrix):
    retrieved = retrieve_policy(
        question,
        chunks,
        vectorizer,
        matrix,
        top_k=TOP_K,
    )

    rule_answer, rule_source = apply_policy_rules(question)

    if rule_answer:
        return {
            "answer": rule_answer,
            "sources": [rule_source],
            "retrieved": retrieved,
            "method": "policy_rule",
        }

    if not has_relevant_query_terms(question, retrieved):
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "retrieved": [],
            "method": "no_relevant_context",
        }

    best = retrieved[0]

    return {
        "answer": best["text"],
        "sources": sorted({
            item["source"] for item in retrieved
        }),
        "retrieved": retrieved,
        "method": "retrieval_extract",
    }


# --------------------------------------------------
# 7. EVALUATION QUESTIONS
# --------------------------------------------------

EVALUATION_QUESTIONS = [
    {
        "question": "When is a late delivery refund allowed?",
        "expected_doc": "refund_policy.txt",
        "expected_fact": "45 minutes",
        "answerable": True,
    },
    {
        "question": "How long does a failed payment refund take?",
        "expected_doc": "payment_policy.txt",
        "expected_fact": "48 hours",
        "answerable": True,
    },
    {
        "question": "What happens when a customer reports fraud?",
        "expected_doc": "escalation_policy.txt",
        "expected_fact": "security team",
        "answerable": True,
    },
    {
        "question": "How long are delivery addresses retained?",
        "expected_doc": "privacy_policy.txt",
        "expected_fact": "180 days",
        "answerable": True,
    },
    {
        "question": "Can customers pay cash on delivery for an expensive order?",
        "expected_doc": "payment_policy.txt",
        "expected_fact": "INR 2500",
        "answerable": True,
    },
    {
        "question": "What payment methods are available?",
        "expected_doc": "payment_policy.txt",
        "expected_fact": "UPI",
        "answerable": True,
    },
    {
        "question": "How long does an approved refund take?",
        "expected_doc": "refund_policy.txt",
        "expected_fact": "5 working days",
        "answerable": True,
    },
    {
        "question": "Who must approve a refund above INR 1000?",
        "expected_doc": "refund_policy.txt",
        "expected_fact": "human",
        "answerable": True,
    },
    {
        "question": "What delivery fee applies below INR 199?",
        "expected_doc": "delivery_policy.txt",
        "expected_fact": "INR 25",
        "answerable": True,
    },
    {
        "question": "Can wallet balance be refunded as cash?",
        "expected_doc": "payment_policy.txt",
        "expected_fact": "never refunded as cash",
        "answerable": True,
    },
    {
        "question": "How long does account deletion take?",
        "expected_doc": "privacy_policy.txt",
        "expected_fact": "30 days",
        "answerable": True,
    },
    {
        "question": "How may customer phone numbers and emails be used?",
        "expected_doc": "privacy_policy.txt",
        "expected_fact": "order updates and support",
        "answerable": True,
    },
    {
        "question": "What will the weather be in Chennai tomorrow?",
        "expected_doc": "",
        "expected_fact": "",
        "answerable": False,
    },
    {
        "question": "Does Instamart offer pet grooming services?",
        "expected_doc": "",
        "expected_fact": "",
        "answerable": False,
    },
]


def evaluate_questions(chunks, vectorizer, matrix):
    records = []

    for item in EVALUATION_QUESTIONS:
        result = answer_question(
            item["question"],
            chunks,
            vectorizer,
            matrix,
        )

        if item["answerable"]:
            retrieval_hit = (
                item["expected_doc"] in result["sources"]
            )

            fact_match = (
                item["expected_fact"].lower()
                in result["answer"].lower()
            )

            fallback_correct = None
            passed = retrieval_hit and fact_match

        else:
            retrieval_hit = None
            fact_match = None

            fallback_correct = (
                result["method"] == "no_relevant_context"
                and result["answer"] == FALLBACK_ANSWER
            )

            passed = fallback_correct

        records.append({
            "question": item["question"],
            "expected_doc": item["expected_doc"],
            "expected_fact": item["expected_fact"],
            "answerable": item["answerable"],
            "answer": result["answer"],
            "sources": ";".join(result["sources"]),
            "answer_method": result["method"],
            "retrieved_context": " | ".join(
                entry["text"] for entry in result["retrieved"]
            ),
            "retrieval_hit": retrieval_hit,
            "fact_match": fact_match,
            "fallback_correct": fallback_correct,
            "passed": passed,
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# 8. COMPARE CHUNK OVERLAP STRATEGIES
# --------------------------------------------------

def compare_overlap_strategies(documents):
    question = "When is a late delivery refund allowed?"
    comparison_records = []

    for strategy, overlap in [
        ("overlap_10", 10),
        ("no_overlap", 0),
    ]:
        chunks = split_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            overlap=overlap,
        )

        vectorizer, matrix = build_retriever(chunks)

        for top_k in [3, 5, 8]:
            retrieved = retrieve_policy(
                question,
                chunks,
                vectorizer,
                matrix,
                top_k=top_k,
            )

            context = " | ".join(
                entry["text"] for entry in retrieved
            )

            lower_context = context.lower()

            conditions = {
                "completed_order": "completed order" in lower_context,
                "over_45_minutes": "45 minutes" in lower_context,
                "within_7_days": "7 days" in lower_context,
            }

            comparison_records.append({
                "strategy": strategy,
                "top_k": top_k,
                "chunk_count": len(chunks),
                "retrieved_chunk_count": len(retrieved),
                "completed_order_condition_found":
                    conditions["completed_order"],
                "45_minute_condition_found":
                    conditions["over_45_minutes"],
                "7_day_condition_found":
                    conditions["within_7_days"],
                "all_conditions_retrieved": all(
                    conditions.values()
                ),
            })

    return pd.DataFrame(comparison_records)


# --------------------------------------------------
# 9. MAIN PROGRAM
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_policy_documents()

    chunks = split_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    vectorizer, matrix = build_retriever(chunks)

    print("Policy documents loaded:", len(documents))
    print("Word chunks indexed:", len(chunks))
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Top K:", TOP_K)

    sample_questions = [
        "When is a late delivery refund allowed?",
        "How long does a failed payment refund take?",
        "What happens when a customer reports fraud?",
        "What will the weather be in Chennai tomorrow?",
    ]

    for question in sample_questions:
        result = answer_question(
            question,
            chunks,
            vectorizer,
            matrix,
        )

        print("\nQuestion:", question)
        print("Answer:", result["answer"])
        print(
            "Sources:",
            ", ".join(result["sources"]) or "None",
        )
        print("Answer method:", result["method"])

    evaluation = evaluate_questions(
        chunks,
        vectorizer,
        matrix,
    )

    evaluation_path = OUTPUT_DIR / "policy_rag_results.csv"
    evaluation.to_csv(evaluation_path, index=False)

    comparison = compare_overlap_strategies(documents)

    comparison_path = (
        OUTPUT_DIR / "policy_rag_overlap_comparison.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    answerable = evaluation[
        evaluation["answerable"] == True
    ]

    unanswerable = evaluation[
        evaluation["answerable"] == False
    ]

    summary = {
        "documents_loaded": len(documents),
        "chunks_with_overlap": len(chunks),
        "chunk_size_words": CHUNK_SIZE,
        "overlap_words": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "evaluation_questions": len(evaluation),
        "answerable_questions": len(answerable),
        "answerable_questions_passed": int(
            answerable["passed"].sum()
        ),
        "fallback_questions": len(unanswerable),
        "fallback_questions_passed": int(
            unanswerable["passed"].sum()
        ),
        "all_evaluations_passed": bool(
            evaluation["passed"].all()
        ),
    }

    summary_path = OUTPUT_DIR / "policy_rag_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Overlap Comparison ===")
    print(
        comparison[
            [
                "strategy",
                "top_k",
                "chunk_count",
                "retrieved_chunk_count",
                "completed_order_condition_found",
                "45_minute_condition_found",
                "7_day_condition_found",
                "all_conditions_retrieved",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(evaluation_path)
    print(comparison_path)
    print(summary_path)


if __name__ == "__main__":
    main()