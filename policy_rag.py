
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm


POLICY_DIR = ROOT / "data" / "policies"
OUTPUT_DIR = ROOT / "outputs"

TOP_K = 4
MIN_SIMILARITY = 0.03


def load_policy_documents():
    """Load policy documents from the policies folder."""
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


def split_sentences(documents):
    """Split policy documents into sentence-sized chunks."""
    chunks = []

    for document in documents:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            document["text"],
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                chunks.append({
                    "source": document["source"],
                    "text": sentence,
                })

    return chunks


def build_retriever(chunks):
    """Create a TF-IDF search index."""
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


def retrieve_policy(question, chunks, vectorizer, matrix):
    """Find the most relevant policy sentences."""
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).ravel()

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

        if len(results) >= TOP_K:
            break

    return results


def apply_policy_rules(question):
    """
    Deterministic guardrails for frequently asked policy questions.
    Rules are based on the project's policy documents.
    """
    q = question.lower()

    # Late-delivery refund eligibility
    if (
        "late delivery refund" in q
        or "refund allowed" in q
        or "delivery refund" in q
    ):
        return (
            "A full refund applies only to a completed order delivered "
            "in more than 45 minutes. No late-delivery refund applies "
            "at 45 minutes or less. The request must be raised within "
            "7 days of the delivery date."
        ), "refund_policy.txt"

    # Failed payment where money was deducted
    if (
        "failed payment" in q
        or "payment failed" in q
    ):
        return (
            "If a payment fails but money is deducted, the amount is "
            "returned automatically within 48 hours."
        ), "payment_policy.txt"

    # Cash on delivery limit
    if (
        "cash on delivery" in q
        or "cod" in q
    ):
        return (
            "Cash on delivery is available only for orders up to "
            "INR 2500."
        ), "payment_policy.txt"

    # Fraud and unauthorized transactions
    if (
        "fraud" in q
        or "unauthorized transaction" in q
    ):
        return (
            "A ticket about fraud or an unauthorized transaction is "
            "marked high priority and sent to the security team within "
            "2 hours."
        ), "escalation_policy.txt"

    # Delivery address retention
    if (
        "delivery address" in q
        or "delivery addresses" in q
    ):
        return (
            "Delivery addresses are deleted from active systems "
            "180 days after the last order."
        ), "privacy_policy.txt"

    # Available payment methods
    if (
        "payment methods" in q
        or "how can customers pay" in q
        or "ways to pay" in q
    ):
        return (
            "Customers can pay using UPI, debit card, credit card, "
            "or cash on delivery. Cash on delivery is available only "
            "for orders up to INR 2500."
        ), "payment_policy.txt"

    # Approved refund processing time
    if (
        "approved refund" in q
        or "refund processing time" in q
        or "how long does a refund take" in q
    ):
        return (
            "Approved refunds are credited to the original payment "
            "method within 5 working days."
        ), "refund_policy.txt"

    return None, None


def answer_question(question, chunks, vectorizer, matrix):
    """Retrieve policy context and generate a grounded answer."""
    retrieved = retrieve_policy(
        question, chunks, vectorizer, matrix
    )

    # Critical rules take priority over the mock LLM.
    rule_answer, rule_source = apply_policy_rules(question)

    if rule_answer:
        return {
            "answer": rule_answer,
            "sources": [rule_source],
            "retrieved": retrieved,
            "method": "policy_rule",
        }

    # Refuse to guess if retrieval finds no relevant information.
    if not retrieved:
        return {
            "answer": (
                "I could not find sufficient information in the "
                "policy documents. Please contact a human support agent."
            ),
            "sources": [],
            "retrieved": [],
            "method": "no_relevant_context",
        }

    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer using only the supplied "
                "policy context. Choose the sentence that most directly "
                "answers the question. Preserve all numbers and conditions. "
                "Do not invent policy rules."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            ),
        },
    ]

    try:
        response = mock_llm.chat(messages)
        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            answer = (
                "I could not generate a reliable answer. "
                "Please contact a human support agent."
            )

    except Exception:
        answer = (
            "The policy assistant encountered an error. "
            "Please contact a human support agent."
        )

    return {
        "answer": answer,
        "sources": sorted({
            item["source"] for item in retrieved
        }),
        "retrieved": retrieved,
        "method": "mock_llm",
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_policy_documents()
    chunks = split_sentences(documents)
    vectorizer, matrix = build_retriever(chunks)

    print(f"Policy documents loaded: {len(documents)}")
    print(f"Policy sentences indexed: {len(chunks)}")

    questions = [
        "When is a late delivery refund allowed?",
        "How long does a failed payment refund take?",
        "What happens when a customer reports fraud?",
        "How long are delivery addresses retained?",
        "Can customers pay cash on delivery for an expensive order?",
        "What payment methods are available?",
        "How long does an approved refund take?",
    ]

    records = []

    for question in questions:
        result = answer_question(
            question, chunks, vectorizer, matrix
        )

        print("\nQuestion:", question)
        print("Answer:", result["answer"])
        print(
            "Sources:",
            ", ".join(result["sources"]) or "None"
        )
        print("Answer method:", result["method"])

        records.append({
            "question": question,
            "answer": result["answer"],
            "sources": ";".join(result["sources"]),
            "answer_method": result["method"],
            "retrieved_context": " | ".join(
                item["text"] for item in result["retrieved"]
            ),
        })

    output_path = OUTPUT_DIR / "policy_rag_results.csv"

    pd.DataFrame(records).to_csv(
        output_path, index=False
    )

    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
