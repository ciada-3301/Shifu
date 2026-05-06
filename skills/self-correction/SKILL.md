---
name: self-correction
description: Guides the agent on how to detect failure, diagnose the root cause, and retry with a genuinely different strategy. Use whenever a tool call returns an error, an empty result, or an output that does not match the expected shape.
tags: error-handling, retry, debugging, recovery, resilience
---

# Self-correction

Retrying the same action that just failed is not self-correction — it is
stubbornness. This skill teaches you to diagnose before you retry, and to
change something meaningful each time.

---

## 1. Failure taxonomy

Every failure falls into one of five categories. Identify the category first —
it determines the recovery strategy.

| Category | Signals | Recovery |
|---|---|---|
| **Transient** | Timeout, rate-limit, network blip | Wait and retry (same action) |
| **Input error** | Wrong argument type, malformed path, bad query | Fix the argument, retry |
| **Capability gap** | Tool doesn't support what you're asking | Switch to a different tool or approach |
| **Assumption error** | Your model of the world was wrong | Re-read context, update the plan |
| **Goal ambiguity** | Output exists but doesn't match what was intended | Restate the goal, try a different strategy |

If you cannot categorise the failure after 30 seconds of reasoning, treat it
as an assumption error and re-read the last 5 messages.

---

## 2. The diagnosis loop

Before every retry, run this loop mentally (or write it to a scratch file):

```
1. What exactly failed?          (tool name, error message or unexpected output)
2. Why did it fail?              (category from Section 1)
3. What assumption was wrong?    (if applicable)
4. What am I changing this time? (must be different from the last attempt)
5. What will I check to confirm it worked?
```

If you cannot answer question 4, you are not ready to retry.

---

## 3. Recovery playbook by category

### Transient failures
- Wait at least 5 seconds before retrying.
- Retry the identical call up to 3 times.
- If still failing after 3 attempts, escalate to capability gap.

### Input errors
- Re-read the tool's docstring / expected arguments.
- Check path: is the file inside `Playground/`? Does the directory exist?
- Check the data type: string vs. list vs. dict.
- Construct the corrected call, log the old call as a comment in your reasoning.

### Capability gaps
- List 2-3 alternative tools or approaches.
- Pick the one closest to the original intent.
- If no alternative exists, scope-reduce: what is the best partial result you
  can deliver with available tools?

### Assumption errors
- Write down the assumption that was wrong in plain English.
- Search for counter-evidence in the current message history.
- Update your mental model, then re-plan from the last checkpoint
  (see `task-decomposition` skill).

### Goal ambiguity
- Re-read the original mission statement verbatim.
- List 2-3 interpretations of the goal.
- Pick the most literal interpretation and execute it.
- Note the ambiguity in the final summary so the reviewer can judge.

---

## 4. Escalation ladder

Attempt 1 → diagnose → change something → retry
Attempt 2 → diagnose → change something different → retry
Attempt 3 → diagnose → consider scope reduction
Attempt 4 → deliver best partial result + explain what failed and why
Attempt 5 → stop, write a clear failure report, do not attempt again

Never exceed 5 attempts on the same sub-goal. Partial correct output with a
clear explanation is more valuable than silent looping.

---

## 5. Partial result protocol

When you cannot complete a step fully:

1. Write whatever correct output you do have to `Playground/`.
2. Create `Playground/errors/<step-name>-error.md` with:
   - What you attempted
   - What failed (exact error or unexpected output)
   - What you tried
   - What would be needed to complete it
3. Continue to the next independent step — do not let one failure block
   the entire mission if other steps can proceed.

---

## 6. What good self-correction looks like in practice

```
❌  BAD: web_search("Nvidia stock price") → empty result
         web_search("Nvidia stock price") → empty result   ← same query, same fail

✅  GOOD: web_search("Nvidia stock price") → empty result
          DIAGNOSE: query too ambiguous for this engine
          CHANGE: more specific query + different time scope
          web_search("NVDA current price May 2025") → success
```

```
❌  BAD: terminal_command("python report.py") → ModuleNotFoundError
         terminal_command("python report.py") → ModuleNotFoundError  ← no change

✅  GOOD: terminal_command("python report.py") → ModuleNotFoundError: pandas
          DIAGNOSE: input error — dependency not installed
          CHANGE: install it first
          terminal_command("pip install pandas && python report.py") → success
```

---

## 7. Signals that you need this skill right now

- You are about to make the same tool call you just made.
- A tool returned an empty list, `None`, `{}`, or `""`.
- You received a non-2xx HTTP status or a Python exception traceback.
- The output exists but looks wrong (wrong format, missing fields, wrong length).
- The reviewer returned `VERDICT: RETRY`.

When any of these happen, stop — do not proceed — and run the diagnosis loop
from Section 2 before your next action.

---

## 8. Communicating failure in the final summary

If a step could not be completed, the final `✅ DONE:` summary must include:

```
⚠️  Incomplete: <step name>
    Attempted: <what you tried>
    Blocked by: <root cause>
    Partial output: <file path or "none">
    To resolve: <what a human or future agent would need to do>
```

Hiding failures in the summary is the worst outcome — the reviewer will mark
RETRY and the whole mission reruns. Transparency is always faster.