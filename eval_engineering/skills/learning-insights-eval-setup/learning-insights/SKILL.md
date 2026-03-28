---
name: learning-insights
description: "Use when the user says /capture-insight, /polish-insight, 'capture this insight', 'save this insight', 'polish this insight', 'create linkedin post', 'turn this into a post', 'this is a good insight', or wants to save a learning moment or convert a saved insight into LinkedIn posts. Triggers on any mention of capturing learnings, saving aha moments, or generating content from past learning sessions."
---

# Learning Insights

Capture raw learning insights from conversations and polish them into LinkedIn posts that sound like the user, not like AI.

## Commands

| Command | Description | Spec |
|---------|-------------|------|
| `/capture-insight <intent>` | Save a raw insight card from the current conversation | `commands/capture-insight.md` |
| `/polish-insight <filename>` | Append LinkedIn post variations to an existing raw insight card | `commands/polish-insight.md` |

**When a command is invoked, read the corresponding file from the `commands/` directory relative to this skill and follow the instructions in it exactly.**

## Storage

All insights are stored in `~/Documents/projects/learning-insights/` (a fixed location, independent of which project you're working in). Insights are organized into project subdirectories based on which project the conversation was happening in.

## Directory Structure

```
~/Documents/projects/learning-insights/
├── INDEX.md                              (global index across all projects)
├── resources/
│   └── voice_samples.md                 (user's actual published LinkedIn posts)
├── ai_engineering/
│   ├── 2026-03-16_recursive-vs-semantic-chunking.md
│   └── ...
├── some_other_project/
│   └── ...
```

**Project detection:** Determine the project name from the current working directory's git repo root folder name. If not in a git repo, use the current directory name. Use this as the subdirectory name under `~/Documents/projects/learning-insights/`.

**Auto-creation:** If `~/Documents/projects/learning-insights/`, the project subdirectory, or `~/Documents/projects/learning-insights/resources/` don't exist, create them automatically.
