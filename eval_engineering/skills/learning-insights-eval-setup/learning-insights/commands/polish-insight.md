# /polish-insight

Take a raw insight card and append polished LinkedIn post variations to the **same file**. The posts must sound like the user — this is the hard part and the whole point.

## Invocation

`/polish-insight <filename or path>`

**Examples:**
- `/polish-insight 2026-03-16_recursive-vs-semantic-chunking.md`
- `/polish-insight ai_engineering/2026-03-16_recursive-vs-semantic-chunking.md`

If just a filename is given, search across all project subdirectories in `~/Documents/projects/learning-insights/`. If ambiguous (same filename in multiple projects), ask the user to specify.

## Process

### 1. Read the raw insight card

Read the file. Check if it already has a "## LinkedIn Post Variations" section. If yes, ask the user if they want to regenerate.

### 2. Study the user's voice

Read `~/Documents/projects/learning-insights/resources/voice_samples.md`.

**If voice samples exist (5+ posts):**
- Analyze: sentence length, vocabulary level, use of questions, paragraph breaks, emoji usage (or absence), how they open posts, how they close posts
- Note their formality level, first person usage, bullet points vs flowing text
- Pay attention to what they DON'T do — no emojis means no emojis, no "Let me explain..." means never use that
- Mirror their patterns confidently

**If fewer than 5 samples or none:**
- Fall back on the user's voice as captured in the raw card (Struggle and Drill-Down sections)
- Keep vocabulary simple and direct
- Avoid LinkedIn AI cliches: "Here's the thing", "Let me break this down", "Game-changer", "Unpopular opinion", "I was today years old", "Stop.", "Read that again."
- Default to short sentences, conversational tone

### 3. Generate 2-3 post variations

Each post MUST:
- Stay under 3000 characters (LinkedIn text post limit)
- Have a strong hook in the first 1-2 lines (visible before "...see more")
- Use line breaks generously — LinkedIn is mobile-first, walls of text get scrolled past
- Sound like a human sharing a genuine learning, not a thought leader performing expertise

**Available formats (pick 2-3 that fit the insight):**

**Narrative / Story:**
Hook (the surprising thing) -> Context/situation -> What you assumed -> What's actually true -> Why it matters -> Takeaway

**Interview Q&A:**
A realistic interview question -> Why it's tricky -> The answer explained clearly -> What this reveals about the concept -> The deeper understanding

**System / Technical breakdown:**
Hook (counterintuitive claim) -> Problem setup -> How it actually works -> The non-obvious detail -> Practical implication

**Myth-buster:**
Common belief -> Why everyone thinks this -> What actually happens -> Evidence/experience -> Corrected mental model

Do NOT force every insight into every format. If only 2 formats work, do 2.

### 4. Append to the same file

Add below the existing raw content. Do NOT modify the raw section.

```markdown

---

## LinkedIn Post Variations

**Polished on:** YYYY-MM-DD
**Voice samples used:** <number of samples available, or "none">

### Variation 1: <format name>

<the actual post content, ready to copy-paste>

---

### Variation 2: <format name>

<the actual post content, ready to copy-paste>

---

### Variation 3: <format name> (if applicable)

<the actual post content, ready to copy-paste>
```

### 5. Update the index

Change Status from `raw` to `polished` for this file's row in `~/Documents/projects/learning-insights/INDEX.md`.

## Voice Matching Rules (non-negotiable)

1. **No vocabulary inflation.** Match the user's reading level. "Use" not "utilize". "Start" not "commence". "Show" not "demonstrate".

2. **No LinkedIn cringe.** Never: one-word-sentence hooks ("Stop."), fake vulnerability, humble-bragging, "agree?", excessive emoji, hashtag walls, tagging random people, "Let that sink in", "Read that again".

3. **Match energy, not just words.** Excited raw insight = excited post. Measured raw insight = measured post.

4. **Genuine hooks only.** First line creates curiosity through: a surprising fact, a counterintuitive claim you can back up, or a question the reader can't immediately answer. Not through clickbait.

5. **End with value, not engagement bait.** Close with the actual takeaway or a thought-provoking implication. Not "What do you think?" or "Drop a comment below."

## Rules

- NEVER add technical claims not in the raw insight. Posts are based on what the user actually learned.
- NEVER make the user sound more expert than the raw card suggests. Learning energy, not retrospective authority.
- If the raw insight is thin (vague struggle, weak aha moment), tell the user the card needs more substance before polishing.
- No placeholder text, no "[insert your experience here]". Everything must be complete and ready to copy-paste.
