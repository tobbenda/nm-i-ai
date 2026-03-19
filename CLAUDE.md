# Project Instructions

## Clarifying questions
- Ask clarifying questions up front, present alternatives in a arrow-selectable menu with a recommended option

## Commit
- Commit after each considerable iteration. Keep message clear and concise. Do not include "co-authored by ...." in the commit message

## Work Doc Logging

After each interaction, append a timestamped log entry to `docs/Work Doc.md` under the `## Log` section.

**Format:** `DD HH:MM - <concise summary of what happened>`

**Examples:**
- `19 14:32 - User asks about concept of batch normalization`
- `19 15:01 - Replaces llama3.2 4b model with llama3.1 8b model`
- `19 15:20 - Initiates error analysis of validation loss spikes`
- `19 15:45 - Decides to use cosine annealing learning rate schedule`

**Rules:**
- Use `date '+%d %H:%M'` shell command to get accurate local DD HH:MM timestamps
- Keep summaries concise — focus on the decision, action, or topic
- Log liberally: any focus shift, exploration of a topic, question about a concept, or decision point deserves an entry
- Only skip truly trivial exchanges (e.g. "ok", "thanks", greetings)
- Append new entries at the bottom of the Log section
- If working with multiple worktrees include (worktree) after the time stamp, before the log entry
