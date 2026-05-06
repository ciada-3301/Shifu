---
name: memory-management
description: Guides the agent on when and how to store, retrieve, compress, and expire information across tool calls and missions. Use when a mission involves recalling past context, building up knowledge over time, avoiding repetition, or managing a long conversation history.
tags: memory, context, recall, compression, summarisation
---

# Memory management

You have a finite context window. Every token spent on stale history is a token
stolen from reasoning. This skill tells you how to keep memory lean, accurate,
and useful.

---

## 1. Classify what needs to be remembered

Before storing anything, decide which bucket it belongs to:

| Type | What it is | Where it lives |
|---|---|---|
| **Working memory** | Facts needed only for this mission | Agent message history (in-context) |
| **Episodic memory** | What happened in past missions | `Playground/memory/episodes.jsonl` |
| **Semantic memory** | Distilled facts about the world / user | `Playground/memory/facts.json` |
| **Procedural memory** | How to do a recurring task | A SKILL.md file |

Do not write to disk for working memory. Do not keep episodic detail in-context
once the mission is done.

---

## 2. When to compress in-context history

Compress when **any** of these are true:
- The message history exceeds ~6000 tokens (roughly 25+ back-and-forth turns).
- Tool results are returning the same information more than once.
- The current step does not depend on the exact wording of earlier steps.

**How to compress:**
1. Call `file_write` to save a one-paragraph summary of what has been
   accomplished so far to `Playground/memory/working_summary.md`.
2. Drop all messages except: the original system prompt, the mission statement,
   and your new summary.
3. Continue from the compressed state.

Never compress the original mission statement. Never compress the most recent
tool result — it is live working memory.

---

## 3. Writing episodic memory (end of mission)

After every COMPLEX mission, append a record to
`Playground/memory/episodes.jsonl`. Each line is a JSON object:

```json
{
  "timestamp": "2025-01-15T14:32:00Z",
  "mission": "one-sentence description of what was asked",
  "outcome": "PASS | FAIL | PARTIAL",
  "key_facts": ["fact 1", "fact 2"],
  "files_created": ["Playground/report.pdf"],
  "tools_used": ["web_search", "terminal_command"],
  "lessons": "one sentence on what to do differently next time"
}
```

Keep `key_facts` to 3 items maximum. Be brutal — only facts that would change
how you approach a similar mission in the future.

---

## 4. Writing and reading semantic facts

Semantic facts are persistent truths about the user, their environment, or their
preferences. They never expire unless explicitly updated.

**Write** a fact when you discover something the user has told you, or something
you have verified:
```json
{
  "key": "user.timezone",
  "value": "Asia/Kolkata",
  "source": "user stated",
  "confidence": "high"
}
```

**Read** facts at the start of any mission that might be affected by them:
```
file_read("memory/facts.json")
```

Update by replacing the value, not appending a duplicate key.

---

## 5. Recall before search

Before calling `web_search` or running a terminal command to find something,
always check:
1. Is it in the current message history?
2. Is it in `Playground/memory/facts.json`?
3. Is there a relevant episode in `Playground/memory/episodes.jsonl`?

If yes — use the cached knowledge. Only search if the information is
time-sensitive (prices, news, live data) or if you have reason to doubt
the cached version.

---

## 6. Memory hygiene rules

- **No duplicate facts.** Before writing a fact, scan for an existing key.
- **No stale episodes.** Episodes older than 30 days are background knowledge,
  not active context. Do not load them unless the mission explicitly refers to
  past work.
- **Compress tool results before storing.** A 4000-token web search result
  becomes a 3-sentence summary in episodic memory.
- **Never store secrets.** API keys, passwords, and personal data do not go
  into any memory file.
- **Playground/memory/ is the only valid memory directory.** Do not scatter
  memory files across the playground root.

---

## 7. Quick reference — which tool to call

| Goal | Tool call |
|---|---|
| Save a working summary mid-mission | `file_write("memory/working_summary.md", ...)` |
| Log a completed mission | `file_write("memory/episodes.jsonl", ...)` (append mode) |
| Store a persistent fact | `file_read` → merge → `file_write("memory/facts.json", ...)` |
| Recall past context | `file_read("memory/episodes.jsonl")` |
| Recall user preferences | `file_read("memory/facts.json")` |