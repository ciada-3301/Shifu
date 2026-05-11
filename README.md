<div align="center">

```
 ███████╗██╗  ██╗██╗███████╗██╗   ██╗
 ██╔════╝██║  ██║██║██╔════╝██║   ██║
 ███████╗███████║██║█████╗  ██║   ██║
 ╚════██║██╔══██║██║██╔══╝  ██║   ██║
 ███████║██║  ██║██║██║     ╚██████╔╝
 ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═════╝
```

**A local AI personal assistant with persistent memory, headless automation, and a rich tool ecosystem.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=flat-square)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What is Shifu?

Shifu is a fully local AI personal assistant that runs in your terminal. It is not a chatbot wrapper — it is a **multi-node agent** built on LangGraph with a rich tool ecosystem, a novel two-tier persistent memory system called SPADA, and a headless automation engine that can schedule and react to events while you are away from your desk.

You talk to Shifu in plain language. Shifu plans, executes, remembers, and automates — all on your own machine.

```
  [10:34]  › remind me to review the PR at 5pm, then send a standup summary to Slack

  ◉ planning…
  ◈ create_automation   ─  "PR review reminder" at cron 0 17 * * *
  ◈ spada_memorise      ─  storing standup preference
  ✓ done in 1.8s

  Automation `pr_review_reminder` registered.
    Trigger  : cron `0 17 * * *`
    Actions  : alarm → slack_message
  The daemon will execute this headlessly.
```

---

## Key Features

- **LangGraph multi-node agent** — planner → executor → reviewer pipeline with per-mission thread isolation and parallel tool execution
- **SPADA** — a custom two-tier memory system (hot JSON + cold ChromaDB) that gives Shifu genuine long-term recall without sending your data anywhere
- **Automator** — a natural-language automation engine backed by a headless daemon: schedule, react to file events, run tool chains, and chain results between steps
- **Companion mode** — lightweight conversational path that bypasses heavy planning for casual or emotional messages
- **Extensive tool suite** — file/directory I/O, browser automation, Google Workspace, weather, alarms, smart home, and more — all auto-discovered at boot
- **Skill system** — drop a `SKILL.md` into `skills/<name>/` and Shifu learns a new capability at next boot
- **Always-on alarm GUI** — Tkinter popup with snooze, countdown, and pulse animation, fired by the daemon
- **Zero cloud dependency** — runs fully offline against any OpenAI-compatible LLM endpoint (Ollama, LM Studio, etc.)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        shifu.py                             │
│   Terminal UI  ──►  LangGraph StateGraph                    │
│                                                             │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐              │
│   │ planner │───►│ executor │───►│ reviewer │              │
│   └─────────┘    └──────────┘    └──────────┘              │
│        │               │                                    │
│        │         ┌─────┴──────┐                            │
│        │         │  tools/    │  (auto-discovered)          │
│        │         └─────┬──────┘                            │
│        │               │                                    │
│   ┌────▼───────────────▼────┐                              │
│   │      SPADA Memory       │                              │
│   │  Hot (JSON)  Cold (DB)  │                              │
│   └─────────────────────────┘                              │
└──────────────────────────────┬──────────────────────────────┘
                               │  subprocess
              ┌────────────────▼────────────────┐
              │         shifu_daemon.py          │
              │   APScheduler + watchdog         │
              │   Executes automations headlessly│
              └─────────────────────────────────┘
```

### Agent pipeline (v5)

| Node | Responsibility |
|---|---|
| **planner** | Single structured LLM call — classifies route (TASK / COMPANION / CLARIFY), decides if memory recall is needed, produces action plan |
| **executor** | Runs tool calls with dependency-aware parallel execution (`asyncio.gather` within a group, sequential across groups) |
| **reviewer** | Fast-path check for obvious errors; full LLM review only when needed |
| **replan** | Re-enters planner after clarification without re-classifying the route (prevents route drift) |

---

## SPADA — Spatial Probability Attention Distribution Algorithm (I developed it)

> SPADA is one of Shifu's two novel core contributions. It gives Shifu a genuine long-term memory that persists across sessions, survives process restarts, and never sends your data to an external service.

SPADA is a **two-tier memory architecture** designed from scratch for Shifu.

### Tier 1 — Hot Memory (zero-latency)

Hot memory is a rolling JSON window stored at `.shifu/hot_memory.json`. After every completed turn, Shifu atomises both the user prompt and its own response into 1–3 key fact strings and appends them to the window.

```
HOT MEMORY — last 3 turns
  T12 [14:22] ← most recent
    IN : user asked to summarise quarterly report PDF
    OUT: summary saved to Playground/q3_summary.md
  T11 [14:18]
    IN : user set location to Kolkata
    OUT: weather tool configured for Kolkata, IST
```

- **Zero lookup cost** — injected directly into every executor system prompt; no embedding call, no vector search
- **Configurable window** — `HOT_MEMORY_MAX_TURNS` env var (default: 15 turns)
- **Intelligent gating** — COMPANION route and trivial queries (≤ 4 words) skip atomisation entirely to avoid wasted spend on chit-chat
- **Automation-aware** — when you create an automation, a compact atom is injected immediately so Shifu can answer *"what automations have I set up?"* without any database lookup
- **Wipe with** `/reset_mem` (cold memory is never touched)

### Tier 2 — Cold Memory (ChromaDB + graph)

Cold memory is a long-term semantic store backed by ChromaDB with a **knowledge graph layer** for concept-linked retrieval.

```
spada_memorise  ──►  atomise text
                ──►  deduplication check (cosine ≥ 0.88 → skip)
                ──►  TTL tagging (session / week / month / permanent)
                ──►  embed with nomic-embed-text-v1.5
                ──►  store in ChromaDB + update graph edges

spada_recall    ──►  embed query
                ──►  ChromaDB similarity search
                ──►  graph expansion (walk edges ≥ 0.75)
                ──►  NeighborMLP re-ranking (3-feature, 2-layer)
                ──►  LLM compresses results to clean prose
                ──►  injected into executor (no raw scores)
```

**Key design properties:**

| Property | Detail |
|---|---|
| **Embeddings** | `nomic-ai/nomic-embed-text-v1.5` via sentence-transformers — runs fully locally |
| **Graph expansion** | NetworkX graph of atom relationships; recalls semantically adjacent facts missed by pure cosine search |
| **NeighborMLP** | Tiny 2-layer MLP (3→8→1) re-ranks graph-expanded candidates using cosine sim, spread score, and graph degree |
| **Deduplication** | Before every write, checks for near-identical existing atoms; skips at ≥ 0.88 similarity |
| **TTL tiers** | `session` (pruned at boot) · `week` · `month` · `permanent` |
| **Clean output** | Raw scored-atom block is never passed to the LLM — only compressed prose summary |
| **NEEDS_MEMORY flag** | Planner explicitly decides YES/NO; pure tool tasks skip recall entirely — zero wasted embedding calls |

**Session close** — on `exit`, `spada_session_close` condenses the session into durable long-term atoms, promoting what matters and discarding noise.

---

## Automator — Natural Language Automation Engine

> The Automator is Shifu's other novel core contribution. It turns a single plain-English sentence into a fully validated, persistently scheduled automation that runs headlessly while Shifu is not even open.

### How it works

```
You say:  "Summarise any PDF dropped in Playground/inbox/ and email me the summary"

Shifu:
  1.  create_automation tool called with your instruction
  2.  Automation LLM structures it into validated JSON
  3.  Schema validation (trigger type, tool names, cron format)
  4.  Written to  .shifu/automations/<id>.yaml
  5.  Hot memory atom injected immediately
  6.  daemon.reload sentinel touched
  7.  Daemon hot-reloads in ≤ 2 seconds
  8.  Watchdog fires on any new PDF in inbox/
  9.  Action chain executes headlessly: read_file → summarise → gmail_send
  10. Result logged to .shifu/automation_log.jsonl
```

### Trigger types

| Trigger | Description | Example |
|---|---|---|
| `schedule` | Standard 5-field cron | `"every weekday at 9am"` |
| `delay` | Relative time, converted to one-shot cron | `"in 20 minutes"` |
| `file_watch` | Fires on file system events | `"when a PDF appears in inbox/"` |
| `startup` | Runs once when daemon next starts | `"on next start, open my dashboard"` |
| `event` | Named event (extensible) | `"when I arrive home"` |

### Action chaining with template variables

Automations support multi-step action chains where the output of one step flows into the next:

```yaml
actions:
  - tool: read_file
    args: { path: "{{trigger_value}}" }
    store_as: file_content

  - tool: summarise
    args: { text: "{{file_content}}", style: brief }
    store_as: summary

  - tool: gmail_send
    args:
      to: you@example.com
      subject: "Summary — {{date}}"
      body: "{{summary}}"
```

**Available template variables:** `{{date}}` · `{{time}}` · `{{trigger_value}}` · `{{previous_result}}` · `{{<store_as_name>}}`

### The Daemon (`shifu_daemon.py`)

The daemon is a standalone process that runs alongside (or independently of) Shifu and executes all automations headlessly.

- **APScheduler** `BackgroundScheduler` — handles all cron/delay triggers
- **watchdog** `ObserverThread` — handles `file_watch` triggers in real time
- **Hot-reload** — polls `daemon.reload` sentinel every 2 seconds; reloads YAML registry without restart
- **Tool registry** — imports the same `tools/` package as Shifu; zero logic duplication
- **Structured logging** — every run (pass or fail) appended to `.shifu/automation_log.jsonl` and `.shifu/automation_results.jsonl`
- **One-shot cleanup** — delay/one-time automations self-delete their YAML after firing
- **PID file** — writes `.shifu/daemon.pid` so Shifu can detect if the daemon is already running

```bash
# Run in foreground
python shifu_daemon.py

# Run silently (warnings and errors only)
python shifu_daemon.py --quiet
```

---

## Tool Ecosystem

Shifu auto-discovers every `BaseTool` instance in the `tools/` package at boot — no manual registration required. Drop a new file in `tools/` and it appears in Shifu's arsenal on next launch.

### Built-in tools

| Category | Tools |
|---|---|
| **Memory** | `spada_recall` · `spada_memorise` · `spada_session_close` |
| **Automation** | `create_automation` |
| **Alarms** | `trigger_alarm` (launches `alarm_gui.py` popup) |
| **File I/O** | `read_file` · `write_file` · `append_file` · `list_directory` · `make_directory` · `delete_file` |
| **Browser** | Custom browser-use implementation — click, scroll, type, screenshot, navigate |
| **Google Workspace** | Gmail send/read · Google Calendar · Google Meet · Google Drive |
| **Weather** | Current conditions, forecast |
| **Smart home** | Device control via `smart_home_set` |
| **Shell** | Execute commands in a sandboxed subprocess |
| **Web search** | Scrape + summarise |

> This list covers the tools visible in the source. The actual tool count is larger — any `BaseTool` in `tools/` is live.

### Skill system

Skills are markdown instruction files that teach Shifu how to handle specific task types without modifying code.

```
skills/
  summarise_legal/
    SKILL.md      ← instructions for summarising legal documents
  python_debugger/
    SKILL.md      ← step-by-step debugging protocol
```

Shifu reads the relevant `SKILL.md` at planning time. List installed skills with `/skills`.

---

## Alarm GUI

When a scheduled alarm fires, the daemon launches `alarm_gui.py` as a subprocess — a standalone Tkinter window that:

- Always-on-top, centred on screen
- Animated pulsing ring icon (orange → bright gold cycle at 40ms)
- Displays label, body message, and fire timestamp
- **Snooze** buttons for 5 / 10 / 15 minutes (re-launches itself after delay)
- **Dismiss** button
- Auto-closes after 5 minutes with a 30-second countdown
- Sound modes: `default` (single bell) · `urgent` (6 bells over 3 seconds) · `silent`

No external dependencies beyond stdlib + tkinter.

---

## Session Awareness

Shifu maintains lightweight cross-mission context within a session via `_SessionContext` — a rolling log of up to 8 recent mission outcomes injected as a slim block into each new executor prompt. This is deliberately not the full message history; it is a distilled thread summary, preventing token bloat while keeping Shifu oriented.

```
══ THIS SESSION ══════════════════════════════════════
Recent missions completed before this one:
  • [10:12] summarized Playground/q3_report.pdf
  • [10:18] created automation pr_review_reminder
  • [10:29] answered weather question for Kolkata
Use this only if the current mission clearly relates to prior work.
═════════════════════════════════════════════════════
```

---

## Getting Started

### Requirements

- Python 3.11+
- An OpenAI-compatible LLM endpoint (Ollama recommended)
- `nomic-embed-text-v1.5` available via sentence-transformers (auto-downloaded on first use)

### Installation

```bash
git clone https://github.com/yourname/shifu.git
cd shifu
pip install -r requirements.txt
```

Core dependencies:

```bash
pip install langchain-core langchain-openai langgraph chromadb \
            sentence-transformers networkx numpy openai pydantic \
            apscheduler watchdog pyyaml python-dotenv tzlocal
```

Optional (for file type support):

```bash
pip install pypdf python-docx openpyxl
```

### Configuration

Create a `.env` file in the project root:

```env
# LLM endpoint (any OpenAI-compatible server)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=llama3:70b          # or any model tag your server serves

# Separate keys per role (optional — falls back to OLLAMA_API_KEY)
OLLAMA_API_KEY_EXECUTOR=...
OLLAMA_API_KEY_PLANNER=...
OLLAMA_API_KEY_AUTOMATION=...

# Memory
SPADA_COLLECTION=shifu_memory
SPADA_PERSIST_DIR=./spada_db_shifu
HOT_MEMORY_MAX_TURNS=15

# Memory tuning (optional)
GRAPH_EDGE_THRESHOLD=0.75
COMPRESS_THRESHOLD=5
DEDUP_THRESHOLD=0.88
```

### Running

```bash
# Start Shifu (daemon auto-spawns)
python shifu.py

# Start the daemon separately (if preferred)
python shifu_daemon.py
python shifu_daemon.py --quiet   # silent mode
```

---

## Terminal Commands

| Command | Description |
|---|---|
| `help` / `?` | Show command reference |
| `history` | Mission log for this session |
| `skills` | List installed skills |
| `files` | Supported file types |
| `mem_status` | Hot + cold memory stats and recent turns |
| `reset_mem` | Wipe hot memory (cold memory untouched) |
| `clear` / `cls` | Reset screen |
| `exit` / `quit` | Shutdown (saves session to cold memory) |
| `<anything else>` | Sent to Shifu as a mission |

---

## Project Structure

```
shifu/
├── shifu.py               # Main agent + terminal UI
├── shifu_daemon.py        # Headless automation daemon
├── tools/
│   ├── __init__.py
│   ├── spada_memory.py    # SPADA memory tools (hot + cold)
│   ├── automator.py       # create_automation tool
│   ├── alarm.py           # trigger_alarm tool
│   └── ...                # All other tools (auto-discovered)
├── alarm_gui.py           # Standalone alarm popup (no deps)
├── skills/                # Drop SKILL.md files here
├── Playground/            # Default working directory
├── .shifu/
│   ├── hot_memory.json    # Rolling hot memory window
│   ├── automations/       # YAML automation specs
│   ├── automation_log.jsonl
│   ├── automation_results.jsonl
│   ├── daemon.pid
│   └── daemon.reload      # Hot-reload sentinel
└── spada_db_shifu/        # ChromaDB vector store
```

---

## Design Principles

**Local first.** All LLM calls go to your own endpoint. Embeddings run locally via sentence-transformers. No data leaves your machine unless a tool explicitly sends it (e.g. Gmail).

**Memory that actually works.** Most agent memory systems are an afterthought. SPADA was designed as a first-class feature: hot memory for immediate continuity, cold memory for long-term recall, graph-expanded retrieval for concept linking, and smart gating so memory lookups only happen when they would actually help.

**Automation as a first-class citizen.** You should be able to say *"do X at Y"* and have it happen without keeping a terminal open. The daemon is a real process with real scheduling, real file watching, and real structured logging — not a cron hack.

**Zero magic registration.** Every tool, skill, and automation is auto-discovered. You extend Shifu by adding files, not by editing registries.

---

## Roadmap

- [ ] Event trigger implementation (location, clipboard, custom webhook)
- [ ] Vision model support for image tools
- [ ] Web UI alongside the terminal interface
- [ ] Automation editor — modify YAML specs via natural language
- [ ] Multi-agent mode — delegate sub-missions to specialised agent instances
- [ ] Mobile notification bridge for alarm and automation results

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

Please make sure new tools follow the `BaseTool` pattern in `tools/` — they will be auto-discovered at boot with no other changes required.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Built with LangGraph · ChromaDB · sentence-transformers · APScheduler · watchdog</sub>
</div>
