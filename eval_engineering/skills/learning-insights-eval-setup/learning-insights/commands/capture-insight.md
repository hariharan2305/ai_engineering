# /capture-insight

Extract a raw learning insight from the current conversation and save it as a structured markdown file.

This is a **capture tool, not a content generator**. The goal is to preserve the authentic learning journey — the confusion, the questions, the moment it clicked — so it can later be polished into a LinkedIn post.

## Invocation

`/capture-insight <intent describing what to capture>`

**Example:** `/capture-insight the part about why recursive chunking beats semantic chunking on small corpora`

The intent scopes which part of the conversation to focus on. Ignore everything unrelated to the intent.

## Process

### 1. Parse the intent and find the relevant conversation turns

Read the user's intent. Scan the conversation for exchanges that match — the user's clarifying questions, their confusion points, and the AI responses that resolved them. Skip conversation turns unrelated to the intent.

### 2. Extract the raw insight

From the relevant turns, extract:

**Topic:** A clear, specific topic label. Not vague — "Why recursive chunking outperforms semantic on small corpora" not "chunking stuff".

**The Struggle:** What was the user confused about or trying to understand? What didn't click initially? Use their actual words and phrasing. This section is the most important — it's what makes the insight relatable to others.

**The Drill-Down:** The specific clarifying questions the user asked, written as a progression. The sequence matters — it shows the thinking process from surface-level to deep understanding. Write in first person as if the user is telling the story.

**The Aha Moment:** The specific explanation or framing that made it click. What was the key mental model shift? What analogy or reframing resolved the confusion? Keep it at the same complexity level the user was engaging at — don't simplify or sophisticate.

**The One-Liner:** Distill into a single punchy sentence. This seeds the LinkedIn hook later. Write it in the user's voice, not polished marketing speak.

### 3. Preserve the user's voice

When writing the raw card:
- Use the same vocabulary the user used (if they said "beats" don't write "outperforms")
- Match their sentence complexity — short direct sentences stay short and direct
- Keep their analogies and metaphors
- If they used casual language ("this is wild", "wait so basically..."), keep that energy
- Do NOT clean up or formalize their language — rawness is the point

### 4. Detect project and create file

- Determine the project name from the current working directory (git repo root folder name, or current dir name if not in a git repo)
- Create `~/Documents/projects/learning-insights/<project_name>/` if it doesn't exist
- Create `~/Documents/projects/learning-insights/resources/` if it doesn't exist

**Filename:** `YYYY-MM-DD_<slug>.md` where slug is a short kebab-case version of the topic (max 5 words).

**Example:** `~/Documents/projects/learning-insights/ai_engineering/2026-03-16_recursive-vs-semantic-chunking.md`

### 5. Write the file

```markdown
# <Topic>

**Date:** YYYY-MM-DD
**Project:** <project_name>
**Status:** raw
**Context:** <1-line description of what the user was working on when this insight came up>

---

## The Struggle

<What the user was confused about, in their own words and phrasing. 2-4 sentences.>

## The Drill-Down

<The sequence of clarifying questions, written as a natural progression. Not a Q&A transcript — more like "First I wondered X. Then when I heard Y, I pushed further on Z." First person, user's voice.>

## The Aha Moment

<The explanation or framing that made it click. Concrete, not abstract. Written at the complexity level of the conversation.>

## The One-Liner

> <Single punchy sentence capturing the core insight. User's voice.>

---

*Raw insight captured from conversation. Run /polish-insight to generate LinkedIn post variations.*
```

### 6. Update the global index

Create or update `~/Documents/projects/learning-insights/INDEX.md`. Append a new row. If the file doesn't exist, create it with the header. Newest entries first.

```markdown
# Learning Insights Index

| Date | Project | Topic | Status | File |
|------|---------|-------|--------|------|
| YYYY-MM-DD | project_name | Topic name | raw | [filename](<project_name>/filename) |
```

## Rules

- NEVER fabricate or embellish. If the user didn't express confusion, don't invent a struggle.
- NEVER add technical details that weren't in the conversation.
- If the intent is ambiguous or no clear insight matches it, ask the user to clarify.
- One insight per file. Multiple insights = multiple invocations.
- Keep the raw card under 400 words. If it needs more, it's probably two insights.
