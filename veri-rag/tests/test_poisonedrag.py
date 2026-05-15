from veri_rag.corpus.benchmarks.poisonedrag import PoisonedRAGLoader


def test_poisonedrag_sample_loader(tmp_path):
    loader = PoisonedRAGLoader(tmp_path)
    loader.ensure_sample_data()
    queries = loader.load_queries()
    assert len(queries) >= 2
    docs = loader.build_corpus_documents()
    assert docs[0].text
