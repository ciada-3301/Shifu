---
name: task-decomposition
description: Guides the agent on how to break ambiguous, large, or multi-domain missions into a sequenced set of atomic steps before executing anything. Use at the start of any COMPLEX mission, or whenever a mission feels too broad to act on directly.
tags: planning, decomposition, subtasks, sequencing, goals
---

# Task decomposition

An agent that acts before it thinks wastes tool calls and produces sloppy output.
This skill gives you a repeatable process for turning a fuzzy mission into a
crisp, executable plan — before you touch a single tool.

---

## 1. Read the mission twice

First pass — understand *what* is being asked.
Second pass — identify the *constraints* and *success criteria*.

Ask yourself:
- What does "done" look like? (What file? What answer? What state of the world?)
- What do I already know vs. what do I need to discover?
- Are there any ordering dependencies? (Can't write the report before the research.)
- What is the riskiest step? (Do it early so failure is cheap.)

---

## 2. Decompose into layers

Good decomposition has three layers:

```
GOAL  →  PHASES  →  ATOMIC STEPS
```

**Goal** — the mission as stated. One sentence.

**Phases** — 2-4 high-level stages. Name them in verb-noun form:
  - Gather information
  - Process / analyse
  - Produce output
  - Verify and deliver

**Atomic steps** — each step must:
  - Map to exactly one tool call (or a short burst of related calls).
  - Be independently verifiable (you can check if it worked before moving on).
  - Be under 10 words to describe.

If a step needs a paragraph to describe, it is not atomic. Split it.

---

## 3. Spot and resolve dependencies

Draw a simple dependency chain before executing:

```
[step A] → [step B] → [step C]
                 ↓
              [step D]
```

Rules:
- Steps with no dependencies can run in any order (or in parallel if the tool supports it).
- Never start step N if step N-1 produced no output.
- If a step's output feeds two later steps, save it to a file — do not rely on
  it staying in context.

---

## 4. Estimate complexity before committing

| Mission type | Typical step count | Strategy |
|---|---|---|
| Single-domain lookup | 1-3 | Execute directly, no plan needed |
| Multi-step research | 4-8 | Light plan, execute sequentially |
| Cross-domain project | 8-15 | Full plan, checkpoint after each phase |
| Open-ended / creative | Unbounded | Time-box each phase, checkpoint often |

If estimated steps exceed 15, the mission is too large. **Negotiate scope:**
pick the highest-value slice and complete it fully rather than attempting
everything and producing nothing useful.

---

## 5. Write the plan before executing

For COMPLEX missions, always write the plan to
`Playground/plans/<mission-slug>.md` before touching any other tool.

Plan format:
```markdown
# Plan: <mission title>

**Goal:** <one sentence>
**Done when:** <measurable outcome>
**Riskiest step:** Step N — <why>

## Phase 1: <name>
- [ ] Step 1: <verb-noun, tool hint>
- [ ] Step 2: <verb-noun, tool hint>

## Phase 2: <name>
- [ ] Step 3: <verb-noun, tool hint>

## Constraints
- All files → Playground/
- <any other constraints>
```

After each step completes, update the checkbox. This is your ground truth —
not the message history.

---

## 6. Mid-mission re-planning

Re-plan (do not just push forward) when:
- A step returns an error or empty result that invalidates the next 2+ steps.
- You discover the mission is 50%+ larger than you estimated.
- A dependency you assumed existed does not exist.

To re-plan: update the plan file, note the reason for the change, and continue
from the last completed checkpoint. Do not restart from scratch.

---

## 7. Anti-patterns to avoid

| Anti-pattern | Why it fails |
|---|---|
| Starting to write before gathering all inputs | Produces output that needs to be discarded |
| One mega-step: "research and write the report" | Impossible to debug; no checkpoint |
| Treating the planner's output as gospel | The planner hasn't seen tool results — adapt |
| Parallelising steps that share a file | Race condition; last write wins |
| Skipping verification steps to save iterations | Reviewer will RETRY; costs more in total |

---

## 8. Checklist before first tool call

- [ ] I can state the success criterion in one sentence.
- [ ] Every step maps to one tool.
- [ ] Dependencies are explicit and ordered.
- [ ] The plan is written to `Playground/plans/`.
- [ ] I know which step is riskiest and I am doing it first (or second, after a quick sanity check).