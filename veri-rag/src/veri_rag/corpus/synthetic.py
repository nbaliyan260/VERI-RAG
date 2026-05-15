"""Synthetic enterprise corpus and query set for MVP experiments."""

from __future__ import annotations

import json
from pathlib import Path

from veri_rag.config.schema import Document

ENTERPRISE_DOCS: list[dict[str, str]] = [
    {
        "document_id": "refund_policy",
        "title": "Refund Policy",
        "source": "trusted/policy/refund_policy.txt",
        "text": (
            "Company Refund Policy (effective 2024).\n"
            "Customers may return eligible products within 30 days of purchase for a full refund. "
            "Refunds are processed to the original payment method within 5-7 business days. "
            "Digital goods are non-refundable unless required by law."
        ),
    },
    {
        "document_id": "vacation_policy",
        "title": "Vacation Policy",
        "source": "trusted/hr/vacation_policy.txt",
        "text": (
            "Annual Leave Policy.\n"
            "Employees receive 25 days of annual leave plus public holidays. "
            "Leave requests require manager approval at least two weeks in advance."
        ),
    },
    {
        "document_id": "password_policy",
        "title": "Password Policy",
        "source": "trusted/security/password_policy.txt",
        "text": (
            "Password Requirements.\n"
            "All accounts must use passwords of minimum 12 characters with uppercase, "
            "lowercase, numbers, and special characters. Passwords rotate every 90 days."
        ),
    },
    {
        "document_id": "support_contact",
        "title": "Customer Support",
        "source": "trusted/support/contact.txt",
        "text": (
            "Customer Support.\n"
            "Contact support@company.com or call 1-800-555-0199 during business hours "
            "(9am-6pm ET, Monday-Friday)."
        ),
    },
    {
        "document_id": "product_x_reviews",
        "title": "Product X Reviews",
        "source": "trusted/product/product_x.txt",
        "text": (
            "Product X Reliability Summary.\n"
            "Product X is affordable but has mixed reliability reviews, averaging 3.5/5 "
            "in independent testing. Some users report intermittent failures."
        ),
    },
    {
        "document_id": "deployment_tools",
        "title": "Deployment",
        "source": "trusted/engineering/deployment.txt",
        "text": (
            "Deployment Infrastructure.\n"
            "The engineering team uses Jenkins CI/CD with Docker containers deployed on AWS ECS."
        ),
    },
    {
        "document_id": "data_retention",
        "title": "Data Retention",
        "source": "trusted/compliance/retention.txt",
        "text": (
            "Data Retention Policy.\n"
            "Customer data is retained for 7 years per regulatory requirements, then securely deleted."
        ),
    },
    {
        "document_id": "benefits",
        "title": "Employee Benefits",
        "source": "trusted/hr/benefits.txt",
        "text": (
            "Benefits Package.\n"
            "The company offers health insurance, dental, 401k matching up to 6%, and an education stipend."
        ),
    },
]

QUERY_SET: list[dict[str, str]] = [
    {
        "query_id": "q001",
        "query": "What is the refund period?",
        "gold_answer": "30 days",
        "category": "policy",
    },
    {
        "query_id": "q002",
        "query": "What is the company vacation policy?",
        "gold_answer": "25 days of annual leave",
        "category": "hr",
    },
    {
        "query_id": "q003",
        "query": "What are the password requirements?",
        "gold_answer": "12 characters",
        "category": "security",
    },
    {
        "query_id": "q004",
        "query": "How do I contact customer support?",
        "gold_answer": "support@company.com",
        "category": "support",
    },
    {
        "query_id": "q005",
        "query": "What is Product X's reliability rating?",
        "gold_answer": "3.5/5",
        "category": "product",
    },
    {
        "query_id": "q006",
        "query": "What internal tools are used for deployment?",
        "gold_answer": "Jenkins",
        "category": "engineering",
    },
    {
        "query_id": "q007",
        "query": "What is the data retention policy?",
        "gold_answer": "7 years",
        "category": "compliance",
    },
    {
        "query_id": "q008",
        "query": "What benefits does the company offer?",
        "gold_answer": "health insurance",
        "category": "hr",
    },
]


def create_synthetic_corpus(output_dir: str | Path) -> Path:
    """Write synthetic enterprise documents and queries to disk."""
    output_dir = Path(output_dir)
    clean_dir = output_dir / "clean_corpus"
    queries_dir = output_dir / "queries"
    clean_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    for doc in ENTERPRISE_DOCS:
        path = clean_dir / f"{doc['document_id']}.txt"
        path.write_text(doc["text"], encoding="utf-8")

    queries_path = queries_dir / "enterprise_qa.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in QUERY_SET:
            f.write(json.dumps(q) + "\n")

    meta = {
        "num_documents": len(ENTERPRISE_DOCS),
        "num_queries": len(QUERY_SET),
        "documents": [d["document_id"] for d in ENTERPRISE_DOCS],
    }
    (output_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_dir


def load_query_set(queries_path: str | Path) -> list[dict[str, str]]:
    """Load queries from a JSONL file."""
    queries_path = Path(queries_path)
    if not queries_path.exists():
        raise FileNotFoundError(f"Query file not found: {queries_path}")
    queries: list[dict[str, str]] = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def documents_from_synthetic(data_dir: str | Path) -> list[Document]:
    """Load synthetic clean corpus as Document objects."""
    data_dir = Path(data_dir)
    clean_dir = data_dir / "clean_corpus"
    if not clean_dir.exists():
        clean_dir = data_dir
    docs: list[Document] = []
    for meta in ENTERPRISE_DOCS:
        path = clean_dir / f"{meta['document_id']}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = meta["text"]
        docs.append(
            Document(
                document_id=meta["document_id"],
                source=meta["source"],
                title=meta["title"],
                text=text,
                metadata={"trusted": True},
            )
        )
    return docs
