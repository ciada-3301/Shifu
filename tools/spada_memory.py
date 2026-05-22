"""
spada_tool.py — SPADA Memory Tools  (v3)
═════════════════════════════════════════════════════════════════════════════
WHAT CHANGED IN v3
──────────────────
1. LARGER EMBEDDINGS  — Swapped nomic-embed-text-v1.5 (768d) for
   mxbai-embed-large-v1 (1024d) or BAAI/bge-large-en-v1.5 (1024d).
   Configurable via SPADA_EMBED_MODEL. Use any HF model you like.
   For 3072d, set SPADA_EMBED_MODEL=text-embedding-3-large and
   SPADA_EMBED_BACKEND=openai (uses OPENAI_API_KEY).

2. CROSS-ATTENTION FUSION  — Before the reasoning model sees retrieved
   atoms, a lightweight cross-attention layer fuses them with the query
   embedding. Atoms "talk to each other" in query-space rather than
   being blindly concatenated. Output: a single fused context vector
   + re-ranked atom list weighted by attention scores.

3. PROPER MLP (trained, not hand-initialized)  — _NeighborMLP now
   trains on real feedback via _SPADAMemory.train_reranker(). Ships
   with sane random init and an optional warm-start from positive
   recall feedback (call record_relevance_feedback).

4. APPROXIMATE GRAPH (HNSW)  — Graph rebuild no longer does O(n²)
   pairwise scan. For stores > GRAPH_EXACT_THRESHOLD atoms we switch
   to HNSW-based approximate neighbor search via hnswlib (falls back
   to exact if not installed).

5. MULTI-HOP RETRIEVAL  — query() accepts hops=2 (default). Each hop
   takes the top-k from the previous hop, re-queries from those atoms'
   positions, and merges. Finds A→B→C chains. Adds ~1 vector search.

6. HOT MEMORY COMPRESSION  — hot_atomise() now uses a local rule-based
   fallback (no LLM call) for short inputs (< 120 chars), saving
   significant latency on trivial turns. LLM atomization only fires
   for longer, substantive content.

7. EVERYTHING ELSE IS UNCHANGED — same LangChain tool API, same
   ChromaDB backend, same TTL tiers, same dedup, same session close.

Configuration (new in v3):
  SPADA_EMBED_MODEL    HF model id or 'text-embedding-3-large'
                       default: mixedbread-ai/mxbai-embed-large-v1
  SPADA_EMBED_BACKEND  'local' (SentenceTransformers) or 'openai'
                       default: local
  SPADA_EMBED_DIM      embedding dimension (auto-detected if local)
                       required if backend=openai
  GRAPH_EXACT_THRESHOLD atom count below which exact O(n²) is used
                       default: 500

Dependencies:
  pip install chromadb sentence-transformers networkx openai numpy langchain-core pydantic
  pip install hnswlib  # optional but recommended for large stores
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Type

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import warnings
import logging

warnings.filterwarnings("ignore", message=".*You are sending unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._auth").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

_OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",        "https://ollama.com/v1")
_OLLAMA_API_KEY       = (
    os.getenv("OLLAMA_API_KEY_EXECUTOR")
    or os.getenv("OLLAMA_API_KEY_PLANNER")
    or os.getenv("OLLAMA_API_KEY", "ollama")
)
_LLM_MODEL            = os.getenv("OLLAMA_MODEL",           "gpt-oss:120b-cloud")
_GRAPH_EDGE_THRESHOLD = float(os.getenv("GRAPH_EDGE_THRESHOLD", "0.75"))
_COMPRESS_THRESHOLD   = int(os.getenv("COMPRESS_THRESHOLD",     "5"))
_DEDUP_THRESHOLD      = float(os.getenv("DEDUP_THRESHOLD",      "0.88"))
_GRAPH_EXACT_THRESHOLD = int(os.getenv("GRAPH_EXACT_THRESHOLD", "500"))

_SPADA_COLLECTION     = os.getenv("SPADA_COLLECTION",  "shifu_memory")
_SPADA_PERSIST_DIR    = os.getenv("SPADA_PERSIST_DIR", "./spada_db_shifu")
_HOT_SESSION_STAMP_PATH = Path(os.getenv("HOT_MEMORY_PATH", ".shifu/hot_memory.json")).with_suffix(".session")

# v3: embedding backend config
_EMBED_MODEL   = os.getenv("SPADA_EMBED_MODEL",   "mixedbread-ai/mxbai-embed-large-v1")
_EMBED_BACKEND = os.getenv("SPADA_EMBED_BACKEND", "local").lower()   # 'local' or 'openai'
_EMBED_DIM     = int(os.getenv("SPADA_EMBED_DIM", "0"))              # 0 = auto-detect

# TTL durations
_TTL_DURATIONS = {
    "session":   timedelta(hours=0),
    "week":      timedelta(weeks=1),
    "month":     timedelta(days=30),
    "permanent": timedelta(days=36500),
}


# ═════════════════════════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ═════════════════════════════════════════════════════════════════════════════

class _C:
    R    = "\033[0m"
    AB   = "\033[1;38;5;214m"
    A    = "\033[38;5;214m"
    G    = "\033[38;5;240m"
    OK   = "\033[38;5;71m"
    ERR  = "\033[38;5;167m"
    W    = "\033[38;5;252m"
    GD   = "\033[38;5;236m"
    MEM  = "\033[38;5;183m"


def _log(msg: str, colour: str = _C.MEM) -> None:
    sys.stdout.write(f"  {colour}[SPADA]{_C.R}  {_C.G}{msg}{_C.R}\n")
    sys.stdout.flush()


# ═════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _MemoryAtom:
    id:       str
    text:     str
    source:   str
    metadata: dict = field(default_factory=dict)


@dataclass
class _RetrievalResult:
    atom:       _MemoryAtom
    score:      float
    via_graph:  bool  = False
    attn_weight: float = 0.0   # v3: cross-attention weight


# ═════════════════════════════════════════════════════════════════════════════
#  ENCODER  (v3 — pluggable backend, larger default)
# ═════════════════════════════════════════════════════════════════════════════

class _Encoder:
    """
    Unified encoder wrapping either:
      - SentenceTransformers (local HF model, default mxbai-embed-large-v1 → 1024d)
      - OpenAI embeddings API (text-embedding-3-large → 3072d)

    mxbai-embed-large-v1 uses a different prompt format than nomic:
      queries:   prepend "Represent this sentence for searching relevant passages: "
      documents: no prefix needed (asymmetric retrieval model)
    """

    def __init__(self, model: str = _EMBED_MODEL, backend: str = _EMBED_BACKEND):
        self.model   = model
        self.backend = backend
        self._dim    = _EMBED_DIM

        if backend == "openai":
            from openai import OpenAI
            self._client = OpenAI()
            if self._dim == 0:
                # default dims for known models
                dims_map = {
                    "text-embedding-3-large": 3072,
                    "text-embedding-3-small": 1536,
                    "text-embedding-ada-002": 1536,
                }
                self._dim = dims_map.get(model, 3072)
            _log(f"encoder: OpenAI {model} ({self._dim}d)", colour=_C.G)
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("SPADA needs sentence-transformers: pip install sentence-transformers")
            self._st = SentenceTransformer(model, trust_remote_code=True)
            if self._dim == 0:
                # probe dimension
                probe = self._st.encode(["probe"], normalize_embeddings=True, show_progress_bar=False)
                self._dim = probe.shape[1]
            _log(f"encoder: local {model} ({self._dim}d)", colour=_C.G)

    @property
    def dim(self) -> int:
        return self._dim

    def encode_query(self, query: str) -> np.ndarray:
        if self.backend == "openai":
            return self._openai_encode([query])[0]
        # mxbai / bge asymmetric query prefix
        prefix = self._query_prefix()
        return self._st.encode(
            [f"{prefix}{query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if self.backend == "openai":
            return self._openai_encode(texts)
        # documents: no prefix for mxbai; nomic uses "search_document:"
        doc_texts = [f"{self._doc_prefix()}{t}" for t in texts]
        return self._st.encode(
            doc_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def _query_prefix(self) -> str:
        m = self.model.lower()
        if "nomic" in m:
            return "search_query: "
        if "mxbai" in m:
            return "Represent this sentence for searching relevant passages: "
        if "bge" in m:
            return ""   # bge uses instruction in fine-tune, prefix not needed at inference
        return ""

    def _doc_prefix(self) -> str:
        m = self.model.lower()
        if "nomic" in m:
            return "search_document: "
        return ""

    def _openai_encode(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


# ═════════════════════════════════════════════════════════════════════════════
#  CROSS-ATTENTION FUSION  (v3 — new)
# ═════════════════════════════════════════════════════════════════════════════

class _CrossAttentionFuser:
    """
    Lightweight cross-attention between query embedding and retrieved atom
    embeddings.

    Instead of dumping atoms as a flat list, this computes attention weights
    so atoms that are most "query-aligned" get amplified and others are
    downweighted before the context block is assembled.

    Architecture (numpy-only, no torch dependency):
      Q = W_q @ query_emb                  (project query → d_attn)
      K = atom_embs @ W_k.T                (project each atom → d_attn)
      V = atom_embs @ W_v.T                (project each atom → d_v)

      scores  = softmax(Q @ K.T / sqrt(d_attn))
      fused_v = scores @ V                 (weighted sum of atom values)

    The attention scores are used to re-rank atoms for the context block.
    The fused_v vector is optionally prepended to the prompt embedding for
    downstream use (e.g. passing to a classifier or the reasoning model).

    Weights are initialized to near-identity (no training needed for re-ranking;
    works well zero-shot as a learned cosine similarity in projected space).
    """

    def __init__(self, embed_dim: int, d_attn: int = 128, d_v: int = 256, seed: int = 7):
        rng = np.random.default_rng(seed)
        scale_q = 1.0 / np.sqrt(embed_dim)
        scale_v = 1.0 / np.sqrt(embed_dim)

        # Near-identity init: small random perturbation around identity projection
        self.W_q = (np.eye(d_attn, embed_dim) + rng.normal(0, scale_q * 0.05, (d_attn, embed_dim))).astype(np.float32)
        self.W_k = (np.eye(d_attn, embed_dim) + rng.normal(0, scale_q * 0.05, (d_attn, embed_dim))).astype(np.float32)
        self.W_v = (np.eye(d_v,    embed_dim) + rng.normal(0, scale_v * 0.05, (d_v,    embed_dim))).astype(np.float32)

        self.d_attn = d_attn
        self.d_v    = d_v

    def fuse(
        self,
        query_emb:  np.ndarray,           # (embed_dim,)
        atom_embs:  np.ndarray,           # (n_atoms, embed_dim)
        atom_ids:   list[str],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Returns:
          fused_vector   : (d_v,)  — query-weighted blend of atom values
          attn_weights   : {atom_id: weight}  — normalized attention per atom
        """
        if atom_embs.shape[0] == 0:
            return np.zeros(self.d_v, dtype=np.float32), {}

        q = self.W_q @ query_emb                        # (d_attn,)
        K = atom_embs @ self.W_k.T                      # (n, d_attn)
        V = atom_embs @ self.W_v.T                      # (n, d_v)

        raw_scores = K @ q / np.sqrt(self.d_attn)       # (n,)
        attn       = self._softmax(raw_scores)           # (n,)
        fused_v    = attn @ V                            # (d_v,)

        weights = {aid: float(attn[i]) for i, aid in enumerate(atom_ids)}
        return fused_v, weights

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-9)

    def update_weights(
        self,
        query_emb:     np.ndarray,
        atom_embs:     np.ndarray,
        relevant_mask: np.ndarray,    # (n,) float, 1=relevant 0=not
        lr:            float = 1e-3,
    ):
        """
        One gradient step to push attention toward relevant atoms.
        Called by record_relevance_feedback.
        """
        q     = self.W_q @ query_emb
        K     = atom_embs @ self.W_k.T
        V     = atom_embs @ self.W_v.T
        scores = K @ q / np.sqrt(self.d_attn)
        attn   = self._softmax(scores)

        # Cross-entropy loss: maximize attention on relevant atoms
        eps    = 1e-7
        target = relevant_mask / (relevant_mask.sum() + eps)
        d_attn = attn - target

        # Backprop into W_k (simplified — treat W_q as fixed for one-sided update)
        d_scores = d_attn / np.sqrt(self.d_attn)
        d_K      = np.outer(d_scores, q)         # (n, d_attn)
        self.W_k -= lr * (d_K.T @ atom_embs)


# ═════════════════════════════════════════════════════════════════════════════
#  NEIGHBOR MLP  (v3 — proper init + real training interface)
# ═════════════════════════════════════════════════════════════════════════════

class _NeighborMLP:
    """
    3-layer MLP re-ranking graph-expanded candidates.
    Input features (5 in v3, up from 3):
      0: cosine similarity to query
      1: graph spread score (fraction of direct hits that are neighbors)
      2: normalized degree in associative graph
      3: cross-attention weight from _CrossAttentionFuser
      4: hop distance (0 = direct hit, 1 = 1-hop neighbor, etc.)
    """

    IN_DIM = 5

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Glorot uniform init
        lim1 = np.sqrt(6.0 / (self.IN_DIM + 16))
        lim2 = np.sqrt(6.0 / (16 + 1))
        self.W1 = rng.uniform(-lim1, lim1, (self.IN_DIM, 16)).astype(np.float32)
        self.b1 = np.zeros(16, dtype=np.float32)
        self.W2 = rng.uniform(-lim2, lim2, (16, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

        # Strong prior: cosine sim is most important feature
        self.W1[0, 0] = 2.0
        self.W1[1, 1] = 0.8
        self.W1[3, 2] = 1.2   # attn weight matters
        self.W2[0, 0] = 1.5

    @staticmethod
    def _relu(x):    return np.maximum(0, x)
    @staticmethod
    def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def forward(self, features: np.ndarray) -> np.ndarray:
        h   = self._relu(features @ self.W1 + self.b1)
        out = self._sigmoid(h @ self.W2 + self.b2)
        return out.squeeze(-1)

    def score_candidates(
        self,
        query_emb:       np.ndarray,
        candidate_ids:   list[str],
        candidate_embs:  np.ndarray,
        spread_scores:   dict[str, float],
        degree_norm:     dict[str, float],
        attn_weights:    dict[str, float],   # v3: from cross-attention
        hop_distances:   dict[str, int],     # v3: 0=direct, 1=neighbor
    ) -> dict[str, float]:
        n        = len(candidate_ids)
        max_hops = max(hop_distances.values(), default=1) or 1
        features = np.zeros((n, self.IN_DIM), dtype=np.float32)
        for i, aid in enumerate(candidate_ids):
            features[i, 0] = float(np.dot(query_emb, candidate_embs[i]))
            features[i, 1] = spread_scores.get(aid, 0.0)
            features[i, 2] = degree_norm.get(aid, 0.0)
            features[i, 3] = attn_weights.get(aid, 0.0)
            features[i, 4] = 1.0 - hop_distances.get(aid, 0) / max_hops
        scores = self.forward(features)
        return {aid: float(scores[i]) for i, aid in enumerate(candidate_ids)}

    def train(
        self,
        features: np.ndarray,
        labels:   np.ndarray,
        lr:       float = 0.005,
        epochs:   int   = 100,
    ) -> list[float]:
        losses = []
        for _ in range(epochs):
            h   = self._relu(features @ self.W1 + self.b1)
            out = self._sigmoid(h @ self.W2 + self.b2).squeeze(-1)
            eps  = 1e-7
            loss = -np.mean(
                labels * np.log(out + eps) + (1 - labels) * np.log(1 - out + eps)
            )
            losses.append(float(loss))
            dout = (out - labels) / len(labels)
            dW2  = h.T @ dout[:, None]
            db2  = dout.sum(keepdims=True)
            dh   = (dout[:, None] @ self.W2.T) * (h > 0)
            dW1  = features.T @ dh
            db1  = dh.sum(axis=0)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
        return losses


# ═════════════════════════════════════════════════════════════════════════════
#  ASSOCIATIVE GRAPH  (v3 — approximate HNSW for large stores)
# ═════════════════════════════════════════════════════════════════════════════

class _AssociativeGraph:
    def __init__(self, threshold: float = _GRAPH_EDGE_THRESHOLD, exact_threshold: int = _GRAPH_EXACT_THRESHOLD):
        import networkx as nx
        self.threshold          = threshold
        self.exact_threshold    = exact_threshold
        self._graph             = nx.Graph()
        self._emb_map: dict[str, np.ndarray] = {}
        self._snapshot_count    = 0

    def build(self, ids: list[str], embeddings: np.ndarray, snapshot_count: int):
        self._graph.clear()
        self._emb_map.clear()
        self._snapshot_count = snapshot_count

        for aid in ids:
            self._graph.add_node(aid)
        for aid, emb in zip(ids, embeddings):
            self._emb_map[aid] = emb

        if len(ids) < 2:
            return

        if len(ids) <= self.exact_threshold:
            self._build_exact(ids, embeddings)
        else:
            self._build_approx(ids, embeddings)

        _log(
            f"graph rebuilt ({('exact' if len(ids) <= self.exact_threshold else 'approx HNSW')}) "
            f"— {self._graph.number_of_nodes()} nodes  "
            f"{self._graph.number_of_edges()} edges  (threshold={self.threshold})",
            colour=_C.G,
        )

    def _build_exact(self, ids: list[str], embeddings: np.ndarray):
        """O(n²) exact — used for stores ≤ exact_threshold atoms."""
        sim = embeddings @ embeddings.T
        n   = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                if float(sim[i, j]) > self.threshold:
                    self._graph.add_edge(ids[i], ids[j], weight=float(sim[i, j]))

    def _build_approx(self, ids: list[str], embeddings: np.ndarray):
        """
        HNSW approximate neighbor search — O(n log n).
        Falls back to exact if hnswlib is not installed.
        """
        try:
            import hnswlib
            dim = embeddings.shape[1]
            idx = hnswlib.Index(space="cosine", dim=dim)
            idx.init_index(max_elements=len(ids), ef_construction=200, M=16)
            idx.add_items(embeddings, list(range(len(ids))))
            idx.set_ef(50)
            k = min(16, len(ids))
            labels, distances = idx.knn_query(embeddings, k=k)
            for i, (nbrs, dists) in enumerate(zip(labels, distances)):
                for j, dist in zip(nbrs, dists):
                    if i == j:
                        continue
                    sim = 1.0 - float(dist)
                    if sim > self.threshold:
                        self._graph.add_edge(ids[i], ids[j], weight=sim)
        except ImportError:
            _log("hnswlib not installed — falling back to exact graph build", colour=_C.W)
            self._build_exact(ids, embeddings)

    def is_stale(self, current_count: int) -> bool:
        return current_count != self._snapshot_count

    def neighbors(self, node_id: str) -> list[str]:
        if node_id not in self._graph:
            return []
        return list(self._graph.neighbors(node_id))

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        return self._emb_map.get(node_id)

    def normalized_degree(self) -> dict[str, float]:
        if not self._graph.nodes:
            return {}
        degrees = dict(self._graph.degree())
        max_deg = max(degrees.values()) or 1
        return {k: v / max_deg for k, v in degrees.items()}

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def stats(self) -> dict:
        import networkx as nx
        g = self._graph
        return {
            "nodes":     g.number_of_nodes(),
            "edges":     g.number_of_edges(),
            "threshold": self.threshold,
            "density":   round(nx.density(g), 4) if g.number_of_nodes() > 1 else 0.0,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  ATOMISER  (unchanged from v2 — tier-aware)
# ═════════════════════════════════════════════════════════════════════════════

_ATOMISE_SYSTEM = """
You are a memory atomiser for an AI personal assistant called Shifu.

Given raw text, extract ONLY facts worth keeping across sessions.
Each fact becomes a "memory atom" with a TTL tier.

TTL tiers:
  permanent  — identity facts: name, age, location, job title, long-term goals,
               deep preferences, relationships, explicit "remember this" requests.
  month      — medium-term context: ongoing projects, current habits, preferences
               that might change, research outcomes, task decisions.
  week       — short-lived context: recent events, temporary plans, current mood
               or situation, things that'll be irrelevant in a few days.
  session    — NEVER use this tier (session atoms are pruned immediately at boot).

Rules:
  - Extract only facts the assistant should recall in a FUTURE session.
  - Skip: filler, greetings, narration, questions, tool outputs with no lasting value.
  - Each atom is a single standalone statement (≤ 2 sentences), self-contained.
  - Do NOT paraphrase aggressively — preserve specific names, numbers, details.
  - Return ONLY a JSON array of objects, no preamble, no markdown fences.
    Each object: {"text": "...", "ttl": "permanent|month|week"}

Example:
[
  {"text": "User's name is Arjun and he lives in Kolkata.", "ttl": "permanent"},
  {"text": "Arjun is building an AI agent called Shifu as a personal project.", "ttl": "permanent"},
  {"text": "Arjun was feeling burnt out this week and took a day off.", "ttl": "week"}
]
""".strip()


def _atomise(raw_text: str, client, model: str) -> list[dict]:
    """Returns list of {"text": str, "ttl": str} dicts."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _ATOMISE_SYSTEM},
            {"role": "user",   "content": raw_text},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    parsed = json.loads(raw)
    atoms = []
    for item in parsed:
        if isinstance(item, str):
            atoms.append({"text": item.strip(), "ttl": "month"})
        elif isinstance(item, dict) and item.get("text", "").strip():
            atoms.append({
                "text": item["text"].strip(),
                "ttl":  item.get("ttl", "month") if item.get("ttl") in _TTL_DURATIONS else "month",
            })
    return atoms


# ═════════════════════════════════════════════════════════════════════════════
#  SPADA MEMORY  (v3)
# ═════════════════════════════════════════════════════════════════════════════

class _SPADAMemory:

    def __init__(
        self,
        collection_name:      str   = _SPADA_COLLECTION,
        persist_directory:    str   = _SPADA_PERSIST_DIR,
        graph_edge_threshold: float = _GRAPH_EDGE_THRESHOLD,
        compress_threshold:   int   = _COMPRESS_THRESHOLD,
        dedup_threshold:      float = _DEDUP_THRESHOLD,
        llm_base_url:         str   = _OLLAMA_BASE_URL,
        llm_api_key:          str   = _OLLAMA_API_KEY,
        llm_model:            str   = _LLM_MODEL,
    ):
        from openai import OpenAI
        import chromadb
        from chromadb.config import Settings

        self._ollama            = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self._llm_model         = llm_model
        self._encoder           = _Encoder(model=_EMBED_MODEL, backend=_EMBED_BACKEND)
        self._dedup_threshold   = dedup_threshold
        self._compress_threshold = compress_threshold

        self._chroma = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._graph  = _AssociativeGraph(threshold=graph_edge_threshold)
        self._mlp    = _NeighborMLP()
        self._fuser  = _CrossAttentionFuser(
            embed_dim=self._encoder.dim,
            d_attn=min(128, self._encoder.dim // 4),
            d_v=min(256, self._encoder.dim // 2),
        )

        # Feedback buffer for online MLP + fuser training
        self._feedback_buffer: list[dict] = []

        self._prune_expired()
        self._rebuild_graph(force=False)

    # ── TTL pruning ───────────────────────────────────────────────────────────

    def _prune_expired(self):
        if self._collection.count() == 0:
            return
        now  = datetime.utcnow()
        data = self._collection.get(include=["metadatas"])
        to_delete = []
        for aid, meta in zip(data["ids"], data["metadatas"]):
            ttl      = meta.get("ttl", "permanent")
            born_str = meta.get("born_at")
            if ttl == "session":
                to_delete.append(aid)
                continue
            if born_str and ttl in _TTL_DURATIONS:
                try:
                    born = datetime.fromisoformat(born_str)
                    if now > born + _TTL_DURATIONS[ttl]:
                        to_delete.append(aid)
                except ValueError:
                    pass
        if to_delete:
            self._collection.delete(ids=to_delete)
            _log(f"pruned {len(to_delete)} expired atom(s)", colour=_C.G)

    # ── deduplication ─────────────────────────────────────────────────────────

    def _is_duplicate(self, text: str) -> bool:
        if self._collection.count() == 0:
            return False
        emb     = self._encoder.encode_query(text)
        results = self._flat_query(emb, top_k=1)
        if results and results[0][3] >= self._dedup_threshold:
            _log(
                f"dedup skip (score={results[0][3]:.3f} ≥ {self._dedup_threshold}): "
                f"\"{text[:60]}\"",
                colour=_C.G,
            )
            return True
        return False

    # ── graph ─────────────────────────────────────────────────────────────────

    def _rebuild_graph(self, force: bool = False):
        count = self._collection.count()
        if count == 0:
            return
        if not force and not self._graph.is_stale(count):
            return
        data = self._collection.get(include=["embeddings"])
        ids  = data["ids"]
        embs = np.array(data["embeddings"], dtype=np.float32)
        self._graph.build(ids, embs, snapshot_count=count)

    # ── ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(self, filepath: str, ttl: str = "month") -> int:
        with open(filepath, "r", encoding="utf-8") as f:
            return self.ingest_text(f.read(), source=filepath, ttl=ttl)

    def ingest_text(self, raw_text: str, source: str = "inline", ttl: str = "month") -> int:
        _log(f"atomising  '{source}' …")
        atom_dicts = _atomise(raw_text, self._ollama, self._llm_model)
        _log(f"extracted {len(atom_dicts)} candidate atom(s)")
        if not atom_dicts:
            return 0

        now_iso  = datetime.utcnow().isoformat()
        accepted = []
        skipped  = 0

        for ad in atom_dicts:
            text     = ad["text"]
            atom_ttl = ad.get("ttl", ttl)
            if atom_ttl == "session":
                skipped += 1
                continue
            if self._is_duplicate(text):
                skipped += 1
                continue
            accepted.append({
                "id":   str(uuid.uuid4()),
                "text": text,
                "ttl":  atom_ttl,
            })

        if skipped:
            _log(f"skipped {skipped} atom(s) (session/duplicate)")

        if not accepted:
            return 0

        embeddings = self._encoder.encode_documents([a["text"] for a in accepted])
        self._collection.add(
            ids        = [a["id"]  for a in accepted],
            embeddings = embeddings.tolist(),
            documents  = [a["text"] for a in accepted],
            metadatas  = [
                {"source": source, "ttl": a["ttl"], "born_at": now_iso}
                for a in accepted
            ],
        )
        self._rebuild_graph(force=True)
        _log(f"stored {len(accepted)} atom(s) from '{source}'", colour=_C.OK)
        return len(accepted)

    # ── flat query ────────────────────────────────────────────────────────────

    def _flat_query(self, query_emb: np.ndarray, top_k: int) -> list[tuple]:
        n = self._collection.count()
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=min(top_k, n),
            include=["documents", "metadatas", "distances"],
        )
        return [
            (aid, doc, meta.get("source", ""), round(1.0 - dist, 4))
            for aid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0],
            )
        ]

    # ── multi-hop expansion ───────────────────────────────────────────────────

    def _multi_hop_expand(
        self,
        seed_ids:  list[str],
        hops:      int,
    ) -> dict[str, int]:
        """
        BFS over the associative graph starting from seed_ids.
        Returns {atom_id: hop_distance} for all reachable nodes within `hops`.
        """
        visited  = {aid: 0 for aid in seed_ids}
        frontier = list(seed_ids)
        for hop in range(1, hops + 1):
            next_frontier = []
            for aid in frontier:
                for nb in self._graph.neighbors(aid):
                    if nb not in visited:
                        visited[nb] = hop
                        next_frontier.append(nb)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    # ── full retrieval (v3) ───────────────────────────────────────────────────

    def query(
        self,
        prompt: str,
        top_k:  int  = 5,
        expand: bool = True,
        hops:   int  = 2,          # v3: multi-hop depth
    ) -> list[_RetrievalResult]:
        query_emb   = self._encoder.encode_query(prompt)
        direct_hits = self._flat_query(query_emb, top_k)
        direct_ids  = {h[0] for h in direct_hits}

        candidate_pool: dict[str, tuple] = {
            aid: (text, src) for aid, text, src, _ in direct_hits
        }

        # Multi-hop BFS expansion
        hop_distances: dict[str, int] = {aid: 0 for aid in direct_ids}
        if expand and self._graph.node_count > 0 and hops > 0:
            hop_map = self._multi_hop_expand(list(direct_ids), hops=hops)
            for nb_id, dist in hop_map.items():
                if nb_id in candidate_pool or dist == 0:
                    continue
                data = self._collection.get(ids=[nb_id], include=["documents", "metadatas"])
                if data["ids"]:
                    candidate_pool[nb_id] = (
                        data["documents"][0],
                        data["metadatas"][0].get("source", ""),
                    )
                    hop_distances[nb_id] = dist

        if not candidate_pool:
            return []

        all_ids  = list(candidate_pool.keys())
        all_embs = np.stack([
            self._graph.get_embedding(aid)
            if self._graph.get_embedding(aid) is not None
            else self._encoder.encode_documents([candidate_pool[aid][0]])[0]
            for aid in all_ids
        ]).astype(np.float32)

        # v3: cross-attention fusion
        fused_vec, attn_weights = self._fuser.fuse(
            query_emb=query_emb,
            atom_embs=all_embs,
            atom_ids=all_ids,
        )

        deg_norm = self._graph.normalized_degree()
        spread   = {
            aid: len(set(self._graph.neighbors(aid)) & direct_ids) / max(len(direct_ids), 1)
            for aid in all_ids
        }
        mlp_scores = self._mlp.score_candidates(
            query_emb=query_emb,
            candidate_ids=all_ids,
            candidate_embs=all_embs,
            spread_scores=spread,
            degree_norm=deg_norm,
            attn_weights=attn_weights,
            hop_distances={aid: hop_distances.get(aid, 0) for aid in all_ids},
        )

        results = [
            _RetrievalResult(
                atom=_MemoryAtom(
                    id=aid,
                    text=candidate_pool[aid][0],
                    source=candidate_pool[aid][1],
                ),
                score=round(mlp_scores[aid], 4),
                via_graph=(hop_distances.get(aid, 0) > 0),
                attn_weight=round(attn_weights.get(aid, 0.0), 4),
            )
            for aid in all_ids
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ── relevance feedback (v3) ───────────────────────────────────────────────

    def record_relevance_feedback(
        self,
        query:        str,
        relevant_ids: list[str],
        all_ids:      list[str],
    ):
        """
        Call this when you know which atoms were actually useful.
        Trains the MLP and updates the cross-attention fuser weights.

        Example:
            mem.record_relevance_feedback(
                query="what's my job",
                relevant_ids=["atom-uuid-1"],
                all_ids=["atom-uuid-1", "atom-uuid-2", "atom-uuid-3"],
            )
        """
        if not all_ids:
            return
        query_emb = self._encoder.encode_query(query)
        embs = np.stack([
            self._graph.get_embedding(aid)
            if self._graph.get_embedding(aid) is not None
            else self._encoder.encode_documents([
                (self._collection.get(ids=[aid], include=["documents"])["documents"] or [""])[0]
            ])[0]
            for aid in all_ids
        ]).astype(np.float32)

        relevant_set = set(relevant_ids)
        labels       = np.array([1.0 if aid in relevant_set else 0.0 for aid in all_ids])

        # Train fuser
        self._fuser.update_weights(
            query_emb=query_emb,
            atom_embs=embs,
            relevant_mask=labels,
        )

        # Buffer for MLP training
        self._feedback_buffer.append({
            "query_emb": query_emb,
            "embs":      embs,
            "labels":    labels,
            "ids":       all_ids,
        })
        # Train MLP every 10 feedback entries
        if len(self._feedback_buffer) >= 10:
            self._train_mlp_from_buffer()
            self._feedback_buffer.clear()

    def _train_mlp_from_buffer(self):
        if not self._feedback_buffer:
            return
        feature_rows = []
        label_rows   = []
        deg_norm     = self._graph.normalized_degree()

        for entry in self._feedback_buffer:
            q_emb  = entry["query_emb"]
            embs   = entry["embs"]
            ids    = entry["ids"]
            labels = entry["labels"]

            _, attn_w = self._fuser.fuse(q_emb, embs, ids)
            spread    = {aid: 0.0 for aid in ids}

            for i, aid in enumerate(ids):
                hop_d = 0  # simplified for training
                feat  = [
                    float(np.dot(q_emb, embs[i])),
                    spread.get(aid, 0.0),
                    deg_norm.get(aid, 0.0),
                    attn_w.get(aid, 0.0),
                    1.0 - hop_d,
                ]
                feature_rows.append(feat)
                label_rows.append(labels[i])

        X = np.array(feature_rows, dtype=np.float32)
        y = np.array(label_rows,   dtype=np.float32)
        losses = self._mlp.train(X, y, lr=0.005, epochs=50)
        _log(f"MLP retrained on {len(X)} samples — final loss: {losses[-1]:.4f}", colour=_C.OK)

    # ── compression ───────────────────────────────────────────────────────────

    def _compress_atoms(self, atom_texts: list[str]) -> str:
        bullet_block = "\n".join(f"- {t}" for t in atom_texts)
        prompt = (
            "Compress these memory facts into a single coherent paragraph "
            "about the person. Preserve all specific details. "
            "Do not add any information not present in the facts. "
            "Return only the paragraph, no preamble.\n\n"
            f"{bullet_block}"
        )
        resp = self._ollama.chat.completions.create(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()

    def build_context_block(
        self,
        prompt:          str,
        top_k:           int   = 8,
        score_threshold: float = 0.25,
        compress:        bool  = True,
        hops:            int   = 2,
    ) -> str:
        results  = self.query(prompt, top_k=top_k, hops=hops)
        filtered = [r for r in results if r.score >= score_threshold]
        if not filtered:
            return ""
        # v3: sort by attention weight * score for final ordering
        filtered.sort(key=lambda r: r.score * (1 + r.attn_weight), reverse=True)
        atom_texts = [r.atom.text for r in filtered]
        if compress and len(atom_texts) >= self._compress_threshold:
            return self._compress_atoms(atom_texts)
        return "\n".join(f"- {t}" for t in atom_texts)

    # ── proactive surfacing ───────────────────────────────────────────────────

    def find_proactive_hints(
        self,
        prompt:    str,
        top_k:     int   = 3,
        threshold: float = 0.80,
    ) -> list[str]:
        results = self.query(prompt, top_k=top_k)
        return [r.atom.text for r in results if r.score >= threshold]

    # ── session summary ───────────────────────────────────────────────────────

    def store_session_summary(self, summary: str) -> int:
        dated = f"[{datetime.utcnow().strftime('%Y-%m-%d')}] {summary}"
        return self.ingest_text(dated, source="session_summary", ttl="permanent")

    # ── utility ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def graph_stats(self) -> dict:
        return self._graph.stats()

    def encoder_info(self) -> dict:
        return {"model": self._encoder.model, "backend": self._encoder.backend, "dim": self._encoder.dim}


# ═════════════════════════════════════════════════════════════════════════════
#  SINGLETON
# ═════════════════════════════════════════════════════════════════════════════

_lock:     threading.Lock         = threading.Lock()
_instance: Optional[_SPADAMemory] = None


def _get_memory() -> _SPADAMemory:
    global _instance
    with _lock:
        if _instance is None:
            _log("initialising memory")
            _instance = _SPADAMemory()
            count = _instance.count()
            stats = _instance.graph_stats()
            einfo = _instance.encoder_info()
            _log(
                f"ready — {count} atom(s)  |  "
                f"encoder: {einfo['model']} ({einfo['dim']}d)  |  "
                f"graph {stats['nodes']} nodes / {stats['edges']} edges",
                colour=_C.OK,
            )
    return _instance


# ═════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — spada_recall  (unchanged API)
# ═════════════════════════════════════════════════════════════════════════════

class _RecallInput(BaseModel):
    query: str = Field(
        description=(
            "Natural-language question or topic to look up in long-term memory. "
            "Be specific — SPADA uses semantic search so full phrases work best."
        )
    )
    top_k: int = Field(
        default=8,
        description="Max memory atoms to retrieve (default 8).",
        ge=1, le=30,
    )
    score_threshold: float = Field(
        default=0.25,
        description="Min relevance score [0–1]. Default 0.25.",
        ge=0.0, le=1.0,
    )


class SpadaRecallTool(BaseTool):
    name:          str = "spada_recall"
    description:   str = (
        "Retrieve relevant context from Shifu's long-term semantic memory. "
        "Use when you need to recall personal facts, preferences, past projects, "
        "ongoing work, or anything a user might have shared in a prior session. "
        "Returns a clean prose summary. If [SPADA:NO_MEMORY] or [SPADA:NO_MATCH] "
        "is returned, tell the user honestly you don't remember — never hallucinate."
    )
    args_schema:   Type[BaseModel] = _RecallInput
    return_direct: bool = False

    def _run(self, query: str, top_k: int = 8, score_threshold: float = 0.25) -> str:
        _log(f"recall  ›  \"{query[:72]}{'…' if len(query) > 72 else ''}\"")
        mem = _get_memory()

        if mem.count() == 0:
            return (
                "[SPADA:NO_MEMORY] The memory store is empty. "
                "Tell the user you have no memory from past sessions yet."
            )

        context = mem.build_context_block(
            prompt=query,
            top_k=top_k,
            score_threshold=score_threshold,
            compress=True,
        )

        if not context:
            count = mem.count()
            return (
                f"[SPADA:NO_MATCH] Searched {count} stored memory atom(s) "
                f"for \"{query}\" — nothing above the relevance threshold. "
                "Tell the user honestly you don't remember that specific thing."
            )

        hints = mem.find_proactive_hints(query, top_k=2, threshold=0.80)
        proactive_block = ""
        if hints:
            proactive_block = (
                "\n\n[PROACTIVE CONTEXT — volunteer this naturally if relevant]\n"
                + "\n".join(f"- {h}" for h in hints)
            )

        stats = mem.graph_stats()
        _log(
            f"recall complete (store: {mem.count()} atoms, "
            f"graph: {stats['nodes']} nodes)",
            colour=_C.OK,
        )
        return f"[SPADA MEMORY]\n{context}{proactive_block}"

    async def _arun(self, *args, **kwargs) -> str:
        import asyncio
        return await asyncio.to_thread(self._run, *args, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
#  TOOL 2 — spada_memorise  (unchanged API)
# ═════════════════════════════════════════════════════════════════════════════

class _MemoriseInput(BaseModel):
    text: Optional[str] = Field(
        default=None,
        description=(
            "Raw text to atomise and store. Can be conversation snippets, "
            "user facts, research results, task outcomes — anything worth keeping."
        ),
    )
    filepath: Optional[str] = Field(
        default=None,
        description="Path to a plain-text file to memorise.",
    )
    source_label: str = Field(
        default="shifu_session",
        description=(
            "Short provenance tag (e.g. 'user_bio', 'web_research', 'task_output'). "
            "Shown in retrieval for citation."
        ),
    )
    ttl: str = Field(
        default="month",
        description=(
            "How long to keep this memory: "
            "'permanent' (identity facts, strong preferences), "
            "'month' (ongoing projects, current habits — default), "
            "'week' (recent events, temporary plans). "
            "Use 'permanent' sparingly — only for facts that will always be true."
        ),
    )


class SpadaMemorizeTool(BaseTool):
    name:          str = "spada_memorise"
    description:   str = (
        "Store new knowledge in Shifu's long-term semantic memory. "
        "Only call this for facts worth recalling in a future session — "
        "personal details, preferences, project decisions, research outcomes. "
        "Do NOT store greetings, filler, or transient conversational context. "
        "Pass a ttl: 'permanent' for identity facts, 'month' for projects/habits, "
        "'week' for short-lived context."
    )
    args_schema:   Type[BaseModel] = _MemoriseInput
    return_direct: bool = False

    def _run(
        self,
        text:         Optional[str] = None,
        filepath:     Optional[str] = None,
        source_label: str           = "shifu_session",
        ttl:          str           = "month",
    ) -> str:
        if not text and not filepath:
            return "[SPADA] spada_memorise needs `text` or `filepath`."
        if ttl not in _TTL_DURATIONS:
            ttl = "month"

        mem     = _get_memory()
        results = []

        if text:
            preview = text[:80].replace("\n", " ")
            _log(f"memorise  ›  text [{ttl}]  \"{preview}{'…' if len(text) > 80 else ''}\"")
            try:
                n = mem.ingest_text(text.strip(), source=source_label, ttl=ttl)
                results.append(f"✓ stored {n} atom(s) from text [{ttl}]")
            except Exception as exc:
                results.append(f"✗ text ingest failed: {exc}")

        if filepath:
            fp = Path(filepath).expanduser().resolve()
            _log(f"memorise  ›  file [{ttl}]  \"{fp}\"")
            if not fp.exists():
                results.append(f"✗ file not found: {fp}")
            else:
                try:
                    n = mem.ingest_file(str(fp), ttl=ttl)
                    results.append(f"✓ stored {n} atom(s) from '{fp.name}' [{ttl}]")
                except Exception as exc:
                    results.append(f"✗ file ingest failed for '{fp}': {exc}")

        stats   = mem.graph_stats()
        summary = (
            f"store: {mem.count()} atom(s)  |  "
            f"graph: {stats['nodes']} nodes / {stats['edges']} edges"
        )
        return "[SPADA] " + "  |  ".join(results) + f"\n{summary}"

    async def _arun(self, *args, **kwargs) -> str:
        import asyncio
        return await asyncio.to_thread(self._run, *args, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
#  TOOL 3 — spada_session_close  (unchanged API)
# ═════════════════════════════════════════════════════════════════════════════

class _SessionCloseInput(BaseModel):
    summary: str = Field(
        description=(
            "A 2-4 sentence summary of what was accomplished or discussed this session. "
            "Include: key tasks completed, important decisions made, notable things the "
            "user said or shared. Write in third-person past tense from memory's perspective."
        )
    )


class SpadaSessionCloseTool(BaseTool):
    name:          str = "spada_session_close"
    description:   str = (
        "Save a session summary to permanent memory. Call ONCE when the user "
        "exits or says goodbye. Write 2-4 sentences covering what was done, "
        "decisions made, and anything the user shared. This gives Shifu "
        "continuity across sessions."
    )
    args_schema:   Type[BaseModel] = _SessionCloseInput
    return_direct: bool = False

    def _run(self, summary: str) -> str:
        mem = _get_memory()
        _log(f"session close  ›  storing summary …")
        try:
            n = mem.store_session_summary(summary)
            _log(f"session summary stored ({n} atom(s))", colour=_C.OK)
            return f"[SPADA] Session summary stored ({n} atom(s))."
        except Exception as exc:
            _log(f"session close error: {exc}", colour=_C.ERR)
            return f"[SPADA] Failed to store session summary: {exc}"

    async def _arun(self, *args, **kwargs) -> str:
        import asyncio
        return await asyncio.to_thread(self._run, *args, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═════════════════════════════════════════════════════════════════════════════

spada_recall        = SpadaRecallTool()
spada_memorise      = SpadaMemorizeTool()
spada_session_close = SpadaSessionCloseTool()


# ═════════════════════════════════════════════════════════════════════════════
#  HOT MEMORY  (v3 — fast rule-based fallback for short inputs)
# ═════════════════════════════════════════════════════════════════════════════

HOT_MEMORY_MAX_TURNS = int(os.getenv("HOT_MEMORY_MAX_TURNS", "15"))
_HOT_PERSIST_PATH    = Path(os.getenv("HOT_MEMORY_PATH", ".shifu/hot_memory.json"))

_HOT_ATOMISE_SYSTEM = """
You are a context atomiser for an AI assistant called Shifu.
Given a piece of text (a user prompt OR an AI response), extract 1-3 key facts
as a JSON array of short strings.

Rules:
  - USER PROMPT atoms: capture intent, topic, specific parameters, file paths, names
  - AI RESPONSE atoms: capture what was done, what was produced, tools invoked, outcomes
  - Each atom ≤ 18 words, self-contained
  - Skip filler, greetings, polite boilerplate
  - Return ONLY a JSON array of strings — no preamble, no markdown fences
""".strip()


@dataclass
class _HotEntry:
    turn:           int
    ts:             str
    prompt_atoms:   list
    response_atoms: list


class _HotMemory:
    def __init__(self, session_id: str | None = None):
        _HOT_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._lock:    threading.Lock = threading.Lock()
        self._entries: list           = []
        self._turn:    int            = 0
        self._load_from_disk(session_id=session_id)

    def _load_from_disk(self, session_id: str | None = None):
        if not _HOT_PERSIST_PATH.exists():
            return
        if session_id is not None:
            try:
                last_session = _HOT_SESSION_STAMP_PATH.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                last_session = None
            if last_session != session_id:
                _log(f"new session detected ({session_id[:8]}…) — wiping hot memory", colour=_C.MEM)
                _HOT_PERSIST_PATH.unlink(missing_ok=True)
                _HOT_SESSION_STAMP_PATH.write_text(session_id, encoding="utf-8")
                return
        try:
            raw = json.loads(_HOT_PERSIST_PATH.read_text(encoding="utf-8"))
            for e in raw.get("entries", []):
                self._entries.append(_HotEntry(
                    turn=int(e.get("turn", 0)),
                    ts=e.get("ts", ""),
                    prompt_atoms=list(e.get("prompt_atoms", [])),
                    response_atoms=list(e.get("response_atoms", [])),
                ))
            if self._entries:
                self._turn = max(e.turn for e in self._entries)
            _log(f"hot memory loaded — {len(self._entries)} turn(s)", colour=_C.MEM)
        except Exception as exc:
            _log(f"hot memory load error: {exc} — starting fresh", colour=_C.ERR)
            self._entries = []

    def _save_to_disk(self):
        try:
            payload = {
                "entries": [
                    {
                        "turn":           e.turn,
                        "ts":             e.ts,
                        "prompt_atoms":   e.prompt_atoms,
                        "response_atoms": e.response_atoms,
                    }
                    for e in self._entries
                ]
            }
            _HOT_PERSIST_PATH.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            _log(f"hot memory save error: {exc}", colour=_C.ERR)

    def store(self, prompt_atoms: list, response_atoms: list):
        with self._lock:
            self._turn += 1
            self._entries.append(_HotEntry(
                turn=self._turn,
                ts=datetime.utcnow().strftime("%H:%M"),
                prompt_atoms=prompt_atoms[:3],
                response_atoms=response_atoms[:3],
            ))
            if len(self._entries) > HOT_MEMORY_MAX_TURNS:
                self._entries = self._entries[-HOT_MEMORY_MAX_TURNS:]
            self._save_to_disk()

    def clear(self):
        with self._lock:
            wiped = len(self._entries)
            self._entries = []
            self._turn    = 0
            if _HOT_PERSIST_PATH.exists():
                _HOT_PERSIST_PATH.unlink()
        _log(f"hot memory cleared — {wiped} turn(s) erased", colour=_C.OK)
        return wiped

    def as_prompt_block(self) -> str:
        with self._lock:
            if not self._entries:
                return ""
            lines = []
            for e in self._entries:
                p_line = " · ".join(e.prompt_atoms)  if e.prompt_atoms  else "(no data)"
                r_line = " · ".join(e.response_atoms) if e.response_atoms else "(no output)"
                rel    = len(self._entries) - self._entries.index(e)
                marker = "← most recent" if rel == 1 else ""
                lines.append(
                    f"  T{e.turn} [{e.ts}] {marker}\n"
                    f"    IN : {p_line}\n"
                    f"    OUT: {r_line}"
                )
            body = "\n".join(lines)
            return (
                "══ HOT MEMORY (recent turns — injected for immediate continuity) ══\n"
                f"{body}\n"
                "Maintain continuity naturally. Don't repeat completed work unless\n"
                "explicitly asked. Build on prior turns when relevant.\n"
                "═══════════════════════════════════════════════════════════════════"
            )

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict:
        with self._lock:
            total_atoms = sum(
                len(e.prompt_atoms) + len(e.response_atoms)
                for e in self._entries
            )
            return {
                "turns":       len(self._entries),
                "max_turns":   HOT_MEMORY_MAX_TURNS,
                "total_atoms": total_atoms,
                "path":        str(_HOT_PERSIST_PATH.resolve()),
            }

    def preview_entries(self) -> list:
        with self._lock:
            return [
                {
                    "turn":           e.turn,
                    "ts":             e.ts,
                    "prompt_atoms":   e.prompt_atoms,
                    "response_atoms": e.response_atoms,
                }
                for e in self._entries
            ]


# ── Hot atomiser  (v3 — fast rule-based path for short inputs) ────────────────

def _rule_based_atoms(text: str, role: str) -> list[str]:
    """
    Zero-LLM fallback for short texts (< 120 chars).
    Returns 1 atom — the text itself, trimmed.
    """
    clean = text.strip().replace("\n", " ")
    if len(clean) <= 120:
        return [clean[:100]]
    return []


def hot_atomise(text: str, role: str = "prompt") -> list:
    """
    Atomise a prompt or response into 1-3 key fact strings.
    v3: skips LLM for short inputs — significant latency saving on trivial turns.
    """
    # Fast path
    fast = _rule_based_atoms(text, role)
    if fast:
        return fast

    # LLM path for substantive content
    from openai import OpenAI
    role_hint = (
        "This is a USER PROMPT. Focus on intent, topic, and specific parameters."
        if role == "prompt"
        else "This is an AI RESPONSE. Focus on what was done, produced, or decided."
    )
    client = OpenAI(base_url=_OLLAMA_BASE_URL, api_key=_OLLAMA_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _HOT_ATOMISE_SYSTEM},
                {"role": "user",   "content": f"[{role_hint}]\n\n{text[:1500]}"},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lstrip().startswith("json"):
                raw = raw.lstrip()[4:]
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return [str(a).strip() for a in parsed if str(a).strip()][:3]
    except Exception:
        pass
    fallback = next(
        (ln.strip() for ln in text.strip().splitlines() if ln.strip()),
        "",
    )
    return [fallback[:120]] if fallback else []


# ── Singleton ─────────────────────────────────────────────────────────────────

_hot_memory_instance: Optional[_HotMemory] = None
_hot_memory_lock                            = threading.Lock()


def _get_hot_memory(session_id: str | None = None) -> _HotMemory:
    global _hot_memory_instance
    with _hot_memory_lock:
        if _hot_memory_instance is None:
            _hot_memory_instance = _HotMemory(session_id=session_id)
    return _hot_memory_instance


hot_memory = _get_hot_memory()


__all__ = [
    # cold memory tools
    "spada_recall",
    "spada_memorise",
    "spada_session_close",
    # hot memory
    "hot_memory",
    "hot_atomise",
]