"""Pinned LOCAL reference token estimates, not attested cloud/billing counts.

No downloads, models or dataset loading. The larger detailed API request limit
is a planning profile, not a resolution of the conflicting model overview.
"""

from functools import lru_cache
import hashlib
import importlib.metadata
from pathlib import Path

from tokenizers import Tokenizer

TOKENIZER_REPO = "Qwen/Qwen3-Reranker-8B"
TOKENIZER_REVISION = "5fa94080caafeaa45a15d11f969d7978e087a3db"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
PROFILE = "qwen3_reference_125pct_plus64_per_pair_v1"


def reference_profile(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require a regular local tokenizer file")
    if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Reference tokenizer hash mismatch")
    return {"path": str(path.resolve()), "sha256": TOKENIZER_SHA256,
            "repo": TOKENIZER_REPO, "revision": TOKENIZER_REVISION, "profile": PROFILE,
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "provider_equivalence_attested": False}


@lru_cache(maxsize=1)
def _tokenizer(path, sha256):
    # ponytail: one cached, hash-bound reference asset; no model/remote-code loader.
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("Reference tokenizer changed while loading")
    tokenizer = Tokenizer.from_str(content.decode("utf-8"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def estimate_input(stage, body, profile):
    """Counts full text, then adds heuristic headroom; never truncate to pass.

    Repeated query AND instruction count once per document. Neither the 25%
    margin nor 64 tokens/pair certifies the cloud tokenizer or provider framing.
    """
    if profile != reference_profile(profile["path"]):
        raise ValueError("Reference tokenizer/profile changed")
    if stage == "embedding":
        query, documents, instruction = body["input"][0], [], ""
        limit = 8192
    elif stage == "rerank":
        query, documents, instruction = body["query"], body["documents"], body["instruct"]
        limit = 120000
    else:
        raise ValueError("Only query embedding and rerank inputs are supported")
    tokenizer = _tokenizer(profile["path"], profile["sha256"])
    counts = [len(row.ids) for row in tokenizer.encode_batch([query, instruction, *documents], add_special_tokens=False)]
    query_tokens, instruction_tokens, *document_tokens = counts
    pairs = max(1, len(document_tokens))
    raw = (query_tokens + instruction_tokens) * pairs + sum(document_tokens)
    planning = (raw * 5 + 3) // 4 + 64 * pairs
    longest = max([query_tokens, *document_tokens])
    return {"kind": "LOCAL_REFERENCE_ESTIMATE_NOT_VENDOR_TOKEN_COUNT", "profile": PROFILE,
            "provider_equivalence_attested": False, "query_tokens": query_tokens,
            "instruction_tokens": instruction_tokens, "document_count": len(documents),
            "document_tokens_sum": sum(document_tokens), "longest_text_tokens": longest,
            "repeated_input_tokens": raw, "planning_tokens": planning,
            "request_reference_limit": limit,
            "within_reference_profile": planning <= limit and (stage == "embedding" or (longest * 5 + 3) // 4 + 64 <= 4000)}
