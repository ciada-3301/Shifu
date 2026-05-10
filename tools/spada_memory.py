"""
spada_tool.py — SPADA Memory Tools for Shifu  (v2)
═════════════════════════════════════════════════════════════════════════════
Drop this single file into Shifu's  tools/  folder.
No other SPADA files needed. No path manipulation. No cross-project imports.

Registers two LangChain tools that Shifu's _load_tools() auto-discovers:

  spada_recall    — semantic search over long-term memory
  spada_memorise  — smart-tiered ingest with dedup + TTL

v2 changes
──────────
• TTL system: atoms tagged session / week / month / permanent.
  Session atoms are pruned at boot; week/month atoms expire automatically.
• Deduplication: before writing, recall is checked — if a near-identical
  atom (score ≥ 0.88) already exists the write is skipped silently.
• Clean recall output: raw scored-atom block is stripped from LLM context;
  only the compressed prose summary is passed to the model.
• Smarter atomiser: guided to tag each atom with a tier.

Configuration (all optional — env vars or the defaults below are fine):
  SPADA_COLLECTION      collection name      (default: shifu_memory)
  SPADA_PERSIST_DIR     ChromaDB folder      (default: ./spada_db_shifu)
  OLLAMA_BASE_URL       Ollama server URL    (default: https://ollama.com/v1)
  OLLAMA_API_KEY / OLLAMA_API_KEY_EXECUTOR / OLLAMA_API_KEY_PLANNER
  OLLAMA_MODEL          model tag            (default: gpt-oss:120b-cloud)
  GRAPH_EDGE_THRESHOLD  cosine sim cutoff    (default: 0.75)
  COMPRESS_THRESHOLD    atom count for LLM   (default: 5)
  DEDUP_THRESHOLD       sim cutoff for skip  (default: 0.88)

Dependencies:
  pip install chromadb sentence-transformers networkx openai numpy langchain-core pydantic
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

_SPADA_COLLECTION     = os.getenv("SPADA_COLLECTION",  "shifu_memory")
_SPADA_PERSIST_DIR    = os.getenv("SPADA_PERSIST_DIR", "./spada_db_shifu")

# TTL durations
_TTL_DURATIONS = {
    "session":   timedelta(hours=0),       # pruned at boot always
    "week":      timedelta(weeks=1),
    "month":     timedelta(days=30),
    "permanent": timedelta(days=36500),    # ~100 years
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
    atom:      _MemoryAtom
    score:     float
    via_graph: bool = False


# ═════════════════════════════════════════════════════════════════════════════
#  NOMIC ENCODER
# ═════════════════════════════════════════════════════════════════════════════

class _NomicEncoder:
    MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("SPADA needs sentence-transformers: pip install sentence-transformers")
        self._model = SentenceTransformer(self.MODEL_ID, trust_remote_code=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([f"search_query: {query}"])[0]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.encode([f"search_document: {t}" for t in texts])


# ═════════════════════════════════════════════════════════════════════════════
#  NEIGHBOR MLP
# ═════════════════════════════════════════════════════════════════════════════

class _NeighborMLP:
    """Tiny 2-layer MLP (3 → 8 → 1) re-ranking graph-expanded candidates."""

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (3, 8)).astype(np.float32)
        self.b1 = np.zeros(8, dtype=np.float32)
        self.W1[0, 0] = 2.0
        self.W1[1, 1] = 1.0
        self.W1[2, 2] = 0.5
        self.W2 = rng.normal(0, 0.1, (8, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
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
        query_emb:      np.ndarray,
        candidate_ids:  list[str],
        candidate_embs: np.ndarray,
        spread_scores:  dict[str, float],
        degree_norm:    dict[str, float],
    ) -> dict[str, float]:
        n        = len(candidate_ids)
        features = np.zeros((n, 3), dtype=np.float32)
        for i, aid in enumerate(candidate_ids):
            features[i, 0] = float(np.dot(query_emb, candidate_embs[i]))
            features[i, 1] = spread_scores.get(aid, 0.0)
            features[i, 2] = degree_norm.get(aid, 0.0)
        scores = self.forward(features)
        return {aid: float(scores[i]) for i, aid in enumerate(candidate_ids)}

    def train(
        self,
        features: np.ndarray,
        labels:   np.ndarray,
        lr:       float = 0.01,
        epochs:   int   = 50,
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
#  ASSOCIATIVE GRAPH
# ═════════════════════════════════════════════════════════════════════════════

class _AssociativeGraph:
    def __init__(self, threshold: float = _GRAPH_EDGE_THRESHOLD):
        import networkx as nx
        self.threshold          = threshold
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
        sim = embeddings @ embeddings.T
        n   = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                if float(sim[i, j]) > self.threshold:
                    self._graph.add_edge(ids[i], ids[j], weight=float(sim[i, j]))
        _log(
            f"graph rebuilt — {self._graph.number_of_nodes()} nodes  "
            f"{self._graph.number_of_edges()} edges  (threshold={self.threshold})",
            colour=_C.G,
        )

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
#  ATOMISER  (v2 — tier-aware)
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
#  SPADA MEMORY  (v2)
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

        self._ollama           = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self._llm_model        = llm_model
        self._encoder          = _NomicEncoder()
        self._dedup_threshold  = dedup_threshold
        self._compress_threshold = compress_threshold

        self._chroma = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._graph = _AssociativeGraph(threshold=graph_edge_threshold)
        self._mlp   = _NeighborMLP()

        # Prune expired atoms on boot before doing anything else
        self._prune_expired()
        self._rebuild_graph(force=False)

    # ── TTL pruning ───────────────────────────────────────────────────────────

    def _prune_expired(self):
        """Delete atoms whose TTL has elapsed. Called once at boot."""
        if self._collection.count() == 0:
            return
        now = datetime.utcnow()
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
                    born  = datetime.fromisoformat(born_str)
                    if now > born + _TTL_DURATIONS[ttl]:
                        to_delete.append(aid)
                except ValueError:
                    pass
        if to_delete:
            self._collection.delete(ids=to_delete)
            _log(f"pruned {len(to_delete)} expired atom(s)", colour=_C.G)

    # ── deduplication ─────────────────────────────────────────────────────────

    def _is_duplicate(self, text: str) -> bool:
        """Return True if a near-identical atom already exists (score ≥ threshold)."""
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

        now_iso   = datetime.utcnow().isoformat()
        accepted  = []
        skipped   = 0

        for ad in atom_dicts:
            text = ad["text"]
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

    # ── full retrieval ────────────────────────────────────────────────────────

    def query(
        self,
        prompt: str,
        top_k:  int  = 5,
        expand: bool = True,
    ) -> list[_RetrievalResult]:
        query_emb   = self._encoder.encode_query(prompt)
        direct_hits = self._flat_query(query_emb, top_k)
        direct_ids  = {h[0] for h in direct_hits}

        candidate_pool: dict[str, tuple] = {
            aid: (text, src, False) for aid, text, src, _ in direct_hits
        }

        if expand and self._graph.node_count > 0:
            for hit_id, _, _, _ in direct_hits:
                for nb_id in self._graph.neighbors(hit_id):
                    if nb_id in candidate_pool:
                        continue
                    data = self._collection.get(ids=[nb_id], include=["documents", "metadatas"])
                    if data["ids"]:
                        candidate_pool[nb_id] = (
                            data["documents"][0],
                            data["metadatas"][0].get("source", ""),
                            True,
                        )

        if not candidate_pool:
            return []

        all_ids  = list(candidate_pool.keys())
        all_embs = np.stack([
            self._graph.get_embedding(aid)
            if self._graph.get_embedding(aid) is not None
            else self._encoder.encode_documents([candidate_pool[aid][0]])[0]
            for aid in all_ids
        ]).astype(np.float32)

        deg_norm = self._graph.normalized_degree()
        spread   = {
            aid: len(set(self._graph.neighbors(aid)) & direct_ids) / max(len(direct_ids), 1)
            for aid in all_ids
        }
        mlp_scores = self._mlp.score_candidates(
            query_emb=query_emb, candidate_ids=all_ids,
            candidate_embs=all_embs, spread_scores=spread, degree_norm=deg_norm,
        )

        results = [
            _RetrievalResult(
                atom=_MemoryAtom(
                    id=aid, text=candidate_pool[aid][0], source=candidate_pool[aid][1]
                ),
                score=round(mlp_scores[aid], 4),
                via_graph=candidate_pool[aid][2],
            )
            for aid in all_ids
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

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
    ) -> str:
        """
        Returns a CLEAN prose context block for the LLM.
        No scores, no atom IDs, no debug metadata — just the signal.
        """
        results  = self.query(prompt, top_k=top_k)
        filtered = [r for r in results if r.score >= score_threshold]
        if not filtered:
            return ""
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
        """
        Return atoms that are highly relevant but weren't explicitly asked for.
        Used to let Shifu volunteer relevant context naturally.
        """
        results = self.query(prompt, top_k=top_k)
        return [r.atom.text for r in results if r.score >= threshold]

    # ── session summary ───────────────────────────────────────────────────────

    def store_session_summary(self, summary: str) -> int:
        """Ingest a session summary as a permanent memory atom."""
        dated = f"[{datetime.utcnow().strftime('%Y-%m-%d')}] {summary}"
        return self.ingest_text(dated, source="session_summary", ttl="permanent")

    # ── utility ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def graph_stats(self) -> dict:
        return self._graph.stats()


# ═════════════════════════════════════════════════════════════════════════════
#  SINGLETON
# ═════════════════════════════════════════════════════════════════════════════

_lock:     threading.Lock          = threading.Lock()
_instance: Optional[_SPADAMemory] = None


def _get_memory() -> _SPADAMemory:
    global _instance
    with _lock:
        if _instance is None:
            _log("initialising memory")
            _instance = _SPADAMemory()
            count = _instance.count()
            stats = _instance.graph_stats()
            _log(
                f"ready — {count} atom(s)  |  "
                f"graph {stats['nodes']} nodes / {stats['edges']} edges",
                colour=_C.OK,
            )
    return _instance


# ═════════════════════════════════════════════════════════════════════════════
#  TOOL 1 — spada_recall
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
    """
    Search Shifu's long-term SPADA memory for facts relevant to a query.

    Returns a clean, compressed prose summary ready to reason over.
    No raw scores or debug metadata are passed to the model.
    """

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

    def _run(
        self,
        query:           str,
        top_k:           int   = 8,
        score_threshold: float = 0.25,
    ) -> str:
        _log(f"recall  ›  \"{query[:72]}{'…' if len(query) > 72 else ''}\"")
        mem = _get_memory()

        if mem.count() == 0:
            return (
                "[SPADA:NO_MEMORY] The memory store is empty. "
                "Tell the user you have no memory from past sessions yet."
            )

        # Get clean prose block (no raw atoms, no scores exposed to LLM)
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

        # Check for proactive hints — relevant things not directly asked for
        hints = mem.find_proactive_hints(query, top_k=2, threshold=0.80)
        proactive_block = ""
        if hints:
            proactive_block = (
                "\n\n[PROACTIVE CONTEXT — volunteer this naturally if relevant]\n"
                + "\n".join(f"- {h}" for h in hints)
            )

        stats  = mem.graph_stats()
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
#  TOOL 2 — spada_memorise
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
    """
    Memorise new knowledge into Shifu's long-term SPADA memory store.

    Content is atomised by the LLM into discrete facts with TTL tiers,
    deduplication is checked before writing, and the associative graph
    is rebuilt — so new knowledge is immediately searchable.

    TIER GUIDE — choose carefully:
      permanent  → name, job, deep preferences, long-term goals, "remember this" requests
      month      → active projects, current habits, research that'll stay relevant
      week       → recent events, this week's mood/context, temporary plans

    Do NOT memorise: greetings, your own narration, tool outputs without lasting
    value, questions you just answered, or anything the user wouldn't want recalled.
    """

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
            _log(
                f"memorise  ›  text [{ttl}]  "
                f"\"{preview}{'…' if len(text) > 80 else ''}\""
            )
            try:
                n = mem.ingest_text(text.strip(), source=source_label, ttl=ttl)
                results.append(f"✓ stored {n} atom(s) from text [{ttl}]")
            except Exception as exc:
                results.append(f"✗ text ingest failed: {exc}")
                _log(f"text ingest error: {exc}", colour=_C.ERR)

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
                    _log(f"file ingest error: {exc}", colour=_C.ERR)

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
#  TOOL 3 — spada_session_close
#  Called by Shifu on exit to store a summary of the session.
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
    """
    Store a permanent summary of the current session into long-term memory.
    Call this exactly once when the user exits or says goodbye.
    """

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
#  HOT MEMORY — Ephemeral rolling context  (prompt + response atoms per turn)
# ═════════════════════════════════════════════════════════════════════════════
#
#  Cold memory (above) persists facts for weeks/months in ChromaDB.
#  Hot memory is different — it stores exactly 2 things per completed turn:
#    1. Atomised version of the user's input prompt
#    2. Atomised version of the final model output + actions
#
#  This rolling window is injected into every executor system prompt so
#  Shifu has immediate, zero-latency awareness of what just happened —
#  no embedding lookup needed.
#
#  Config env vars:
#    HOT_MEMORY_MAX_TURNS   max turns to keep          (default: 15)
#    HOT_MEMORY_PATH        path to backing JSON file  (default: .shifu/hot_memory.json)
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

Examples:
  User prompt  → ["User asked for a Python web scraper targeting example.com", "User wants pagination and a CSV output"]
  AI response  → ["Wrote Playground/scraper.py using BeautifulSoup", "Ran pip install requests beautifulsoup4", "Saved 42 results to Playground/results.csv"]
""".strip()


@dataclass
class _HotEntry:
    turn:           int
    ts:             str   # HH:MM wall-clock at storage time
    prompt_atoms:   list
    response_atoms: list


class _HotMemory:
    """
    Rolling per-turn hot memory store.

    Stores (prompt_atoms, response_atoms) pairs for the last N turns and
    surfaces them as a compact plain-text block for system prompt injection.
    Backed by a lightweight JSON file entirely separate from the cold ChromaDB
    store — your spada_db_shifu data is never touched.

    Thread-safe. Wipe with .clear() or the /reset_mem terminal command.
    """

    def __init__(self):
        _HOT_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._lock:    threading.Lock  = threading.Lock()
        self._entries: list            = []   # list[_HotEntry]
        self._turn:    int             = 0
        self._load_from_disk()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_from_disk(self):
        if not _HOT_PERSIST_PATH.exists():
            return
        try:
            raw  = json.loads(_HOT_PERSIST_PATH.read_text(encoding="utf-8"))
            for e in raw.get("entries", []):
                self._entries.append(_HotEntry(
                    turn=int(e.get("turn", 0)),
                    ts=e.get("ts", ""),
                    prompt_atoms=list(e.get("prompt_atoms", [])),
                    response_atoms=list(e.get("response_atoms", [])),
                ))
            if self._entries:
                self._turn = max(e.turn for e in self._entries)
            _log(
                f"hot memory loaded — {len(self._entries)} turn(s) "
                f"(max {HOT_MEMORY_MAX_TURNS})",
                colour=_C.MEM,
            )
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

    # ── write ─────────────────────────────────────────────────────────────────

    def store(self, prompt_atoms: list, response_atoms: list):
        """
        Add a new (prompt, response) atom pair.
        Trims the window to HOT_MEMORY_MAX_TURNS and persists to disk.
        """
        with self._lock:
            self._turn += 1
            self._entries.append(_HotEntry(
                turn=self._turn,
                ts=datetime.utcnow().strftime("%H:%M"),
                prompt_atoms=prompt_atoms[:3],    # hard cap: 3 atoms each
                response_atoms=response_atoms[:3],
            ))
            if len(self._entries) > HOT_MEMORY_MAX_TURNS:
                self._entries = self._entries[-HOT_MEMORY_MAX_TURNS:]
            self._save_to_disk()

    # ── wipe ─────────────────────────────────────────────────────────────────

    def clear(self):
        """Wipe all hot memory. Called by /reset_mem."""
        with self._lock:
            wiped = len(self._entries)
            self._entries = []
            self._turn    = 0
            if _HOT_PERSIST_PATH.exists():
                _HOT_PERSIST_PATH.unlink()
        _log(f"hot memory cleared — {wiped} turn(s) erased", colour=_C.OK)
        return wiped

    # ── read ─────────────────────────────────────────────────────────────────

    def as_prompt_block(self) -> str:
        """
        Returns a compact hot memory injection block for executor system prompts.
        Empty string if no entries yet.
        """
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
        """Return entries as plain dicts for display purposes."""
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


# ── Standalone hot atomiser ───────────────────────────────────────────────────

def hot_atomise(text: str, role: str = "prompt") -> list:
    """
    Atomise a prompt or response into 1-3 key fact strings.
    Uses the same Ollama endpoint as the cold memory LLM.

    role: "prompt"   — user input (extract intent + key details)
          "response" — AI output  (extract what was done/produced)

    Returns a list[str] of atoms. Falls back gracefully on LLM errors.
    """
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
                {
                    "role":    "user",
                    "content": f"[{role_hint}]\n\n{text[:1500]}",
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.lstrip().startswith("json"):
                raw = raw.lstrip()[4:]
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return [str(a).strip() for a in parsed if str(a).strip()][:3]
    except Exception:
        pass
    # Graceful fallback: first non-empty line truncated
    fallback = next(
        (ln.strip() for ln in text.strip().splitlines() if ln.strip()),
        "",
    )
    return [fallback[:120]] if fallback else []


# ── Singleton ─────────────────────────────────────────────────────────────────

_hot_memory_instance: Optional[_HotMemory] = None
_hot_memory_lock                            = threading.Lock()


def _get_hot_memory() -> _HotMemory:
    global _hot_memory_instance
    with _hot_memory_lock:
        if _hot_memory_instance is None:
            _hot_memory_instance = _HotMemory()
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