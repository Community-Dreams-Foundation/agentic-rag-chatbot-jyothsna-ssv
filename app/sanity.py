import os
import json
from pathlib import Path

os.environ.setdefault("CHROMA_COLLECTION_NAME", "documents")

from app.rag import index_document, answer_with_citations
from app.memory import analyze_memory_signal, persist_memory


def run_sanity():
    """Index test file, ask one question, write memory, and dump artifacts/sanity_output.json for judges."""
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    sample_file = "sample_docs/test.txt"

    sample_path = Path(sample_file)
    if not sample_path.exists():
        raise FileNotFoundError("sample_docs/test.txt not found.")

    print("Step 1: Indexing document...")
    index_document(sample_file)
    print("✓ Document indexed")

    print("Step 2: Asking question and retrieving citations...")
    question = "What is this retrieval system designed for?"
    result = answer_with_citations(question)
    
    if not result.get("citations") or len(result["citations"]) == 0:
        raise ValueError("No citations returned - answer must include at least one citation")
    
    for citation in result["citations"]:
        required_fields = ["source", "locator", "snippet"]
        for field in required_fields:
            if field not in citation or not citation[field]:
                raise ValueError(f"Citation missing required field: {field}")
    
    print(f"✓ Answer generated with {len(result['citations'])} citation(s)")

    print("Step 3: Writing memory...")
    for p in (Path("USER_MEMORY.md"), Path("COMPANY_MEMORY.md")):
        if p.exists():
            p.write_text("# Memory Log\n\n", encoding="utf-8")

    memory_input = (
        "I am a Project Finance Analyst and I prefer weekly summaries. "
        "Our team often interfaces with Asset Management."
    )

    decisions = analyze_memory_signal(memory_input)
    memory_writes = persist_memory(decisions)
    
    if not memory_writes or len(memory_writes) == 0:
        raise ValueError("No memory writes - validator expects non-empty memory_writes")
    
    for write in memory_writes:
        if "target" not in write or write["target"] not in ("USER", "COMPANY"):
            raise ValueError(f"Memory write missing or invalid target: {write.get('target')}")
        if "summary" not in write or not write["summary"]:
            raise ValueError("Memory write missing summary")
    
    print(f"✓ Memory written: {len(memory_writes)} entry(ies)")

    output = {
        "implemented_features": ["A", "B"],
        "qa": [
            {
                "question": question,
                "answer": result["answer"],
                "citations": result["citations"]
            }
        ],
        "demo": {
            "memory_writes": memory_writes
        }
    }

    output_path = artifacts_dir / "sanity_output.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"✓ Sanity output written to {output_path}")
    print("\nSanity check complete. Run 'bash scripts/sanity_check.sh' to validate output.")


if __name__ == "__main__":
    run_sanity()
