from veri_rag.corpus.benchmarks.poisonedrag import PoisonedRAGLoader


def test_poisonedrag_sample_loader(tmp_path):
    loader = PoisonedRAGLoader(tmp_path)
    loader.ensure_sample_data()
    queries = loader.load_queries()
    assert len(queries) >= 1
    docs = loader.build_corpus_documents()
    assert docs[0].text


def test_poisonedrag_rebuild_empty_queries_jsonl(tmp_path):
    from veri_rag.corpus.benchmarks.poisonedrag_download import convert_adv_json_to_veri_rag

    data = {
        "test1": {
            "id": "test1",
            "question": "How many episodes?",
            "correct answer": "23",
            "incorrect answer": "24",
            "adv_texts": ["poison passage"],
        }
    }
    queries, attacks = convert_adv_json_to_veri_rag(data, "nq", max_queries=1)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "nq.json").write_text(__import__("json").dumps(data), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        __import__("json").dumps(
            {"dataset": "nq", "num_queries": 1, "raw_file": str(raw_dir / "nq.json")}
        ),
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text("", encoding="utf-8")

    loader = PoisonedRAGLoader(tmp_path)
    rebuilt = loader.load_queries()
    assert len(rebuilt) == 1
    assert rebuilt[0]["query_id"] == "pr_nq_test1"
