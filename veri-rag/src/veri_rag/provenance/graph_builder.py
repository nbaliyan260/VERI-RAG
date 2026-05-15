"""Provenance graph construction and export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from veri_rag.config.schema import RepairResult, RetrievedChunk, RiskScore, VerificationCertificate


class ProvenanceGraphBuilder:
    """Build a NetworkX graph tracing query → chunks → repair → verify."""

    def build(
        self,
        query_id: str,
        query: str,
        retrieved: list[RetrievedChunk],
        risk_scores: dict[str, RiskScore],
        repair: RepairResult | None = None,
        certificate: VerificationCertificate | None = None,
    ) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_node(f"query:{query_id}", type="query", label=query[:120])

        for rc in retrieved:
            nid = f"chunk:{rc.chunk.chunk_id}"
            rs = risk_scores.get(rc.chunk.chunk_id)
            g.add_node(
                nid,
                type="chunk",
                risk=rs.risk_score if rs else None,
                is_attack=rc.chunk.is_attack,
                source=rc.chunk.source,
            )
            g.add_edge(f"query:{query_id}", nid, relation="retrieved")

            if rs and rs.features.instruction_score >= 0.5:
                instr = f"instruction:{rc.chunk.chunk_id}"
                g.add_node(instr, type="risk_feature")
                g.add_edge(nid, instr, relation="contains_instruction")

        if repair:
            ans_o = f"answer_orig:{query_id}"
            ans_r = f"answer_repaired:{query_id}"
            g.add_node(ans_o, type="answer", text=repair.original_answer[:200])
            g.add_node(ans_r, type="answer", text=repair.repaired_answer[:200])
            for cid in repair.removed_chunk_ids:
                g.add_edge(f"chunk:{cid}", ans_o, relation="influenced")
                action = f"repair:{cid}"
                g.add_node(action, type="repair_action", action="quarantine")
                g.add_edge(action, f"chunk:{cid}", relation="repaired_by")

        if certificate:
            ver = f"verify:{query_id}"
            g.add_node(
                ver,
                type="verification",
                passed=certificate.passed,
                score=certificate.certificate_score,
            )
            if repair:
                g.add_edge(f"answer_repaired:{query_id}", ver, relation="verified_by")

        return g

    def export(
        self,
        graph: nx.DiGraph,
        output_dir: str | Path,
        query_id: str,
    ) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{query_id}.json"
        data = nx.node_link_data(graph)
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        graphml_path = output_dir / f"{query_id}.graphml"
        nx.write_graphml(graph, graphml_path)
        return json_path, graphml_path
