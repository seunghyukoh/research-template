---
name: anti-ai-writing
description: >
  Skill for writing English text that reads as naturally human-authored, and for reviewing
  existing English text to detect and fix AI writing tells. Use this skill whenever the user
  asks to "write naturally", "remove AI tone", "make it sound human", "avoid AI tells",
  "de-AI", "humanize", "anti-AI writing", "sound less like ChatGPT", or any variation
  implying they want text free of detectable AI patterns. Also trigger when the user
  asks to review or edit text for AI-sounding patterns, or when they paste text and ask
  "does this sound like AI?" Even if the user simply says "write me a blog post" or
  "write an essay" without explicitly mentioning AI — if this skill is available, apply
  its principles by default to produce more natural prose.
---

# Anti-AI Writing Skill

Two modes: **Write** (produce new text that reads human) and **Review** (audit existing text for AI tells and fix them). Both draw on the AI writing pattern catalogue below, compiled from Wikipedia editors who reviewed thousands of AI-generated texts.

The fundamental insight: AI writing fails not because of any single phrase, but because of **pattern stacking** — multiple small tells co-occurring in predictable combinations. A single "moreover" is fine. "Moreover" + em dash + inflated significance + uniform paragraph length + "-ing" tailing clause = unmistakably AI.

---

## Mode A: Write (Producing New Text)

When writing new text, internalize and apply every guideline below **silently**. Do not mention this skill, do not add disclaimers about "writing naturally", do not call attention to the process. Just write well.

### Core Principles

1. **Say what you mean, then stop.** AI pads sentences with significance claims. Humans state facts and move on. If a sentence works without "highlighting the importance of", cut it.

2. **Vary your rhythm.** AI defaults to uniform sentence and paragraph lengths. Mix short punchy sentences with longer complex ones. Let some paragraphs be two sentences. Let others be five.

3. **Be specific, not sweeping.** AI generalizes ("plays a vital role in the broader landscape"). Humans cite particulars ("cut processing time from 14 hours to 90 minutes").

4. **Earn your transitions.** Don't sprinkle "Moreover" / "Furthermore" / "Additionally" like punctuation. If the next sentence logically follows, you often need no transition at all.

5. **Let the reader draw conclusions.** AI tells readers what to think ("This represents a significant advancement"). Present the evidence and trust the reader.

6. **Write with a point of view.** AI hedges everything into mush. If you're writing an argument, commit to it. Neutral doesn't mean lifeless.

### Positive Habits

- Use contractions where tone permits ("it's", "don't", "can't")
- Use concrete nouns and active verbs over abstract nominalizations
- Start some sentences with "And" or "But"
- Occasionally break a "rule" — fragment sentences, one-word paragraphs, colloquialisms — where it serves the voice
- Reference specifics: names, numbers, dates, places. Not vague gestures at importance.
- When you genuinely need an em dash, use one. But check whether a comma or parenthetical would work just as well.

---

## Mode B: Review (Auditing Existing Text)

When the user provides text for review:

### Step 1: Read the Full Text
Read everything before commenting. Pattern stacking matters more than individual tells.

### Step 2: Assess Overall AI-Signature Density
- **Low**: A few minor tells; would pass casual reading
- **Medium**: Noticeable pattern clustering in some sections
- **High**: Multiple overlapping tells throughout; reads as clearly AI-generated

### Step 3: Flag Specific Issues
For each issue: quote the passage, name the tell category, suggest a concrete rewrite. Group by severity — lead with the strongest AI signals.

### Step 4: Provide a Rewritten Version
Offer a complete rewrite preserving meaning while eliminating AI tells. Don't just swap synonyms — rethink sentence construction, rhythm, and structure.

### Step 5: Summary
Brief count of issues per category, plus 2-3 highest-priority habits for this writer.

---

## AI Writing Tells — Full Catalogue

### 1. Inflated Significance & Symbolism

AI reflexively puffs up the importance of whatever it's writing about.

**Flagged phrases**: "a testament to", "stands as a [symbol/beacon]", "serves as a reminder of", "plays a vital/significant/crucial role in", "underscores its importance", "continues to captivate", "leaves a lasting impact/legacy", "watershed moment", "pivotal turning point", "deeply rooted in", "profound heritage", "steadfast dedication", "solidifies [someone's] position as", "marking a pivotal moment in the evolution of...", "part of a broader movement"

**Example** (real AI Wikipedia text):
> "The Statistical Institute of Catalonia was officially established in 1989, marking a pivotal moment in the evolution of regional statistics in Spain."

**Fix**: State the fact. Let the reader decide its importance. "The Statistical Institute of Catalonia was established in 1989" is sufficient.

### 2. Promotional & Travel-Copy Tone

AI writes about places and cultures as if producing marketing copy.

**Flagged phrases**: "rich cultural heritage", "rich history", "breathtaking", "must-visit / must-see", "stunning natural beauty", "enduring/lasting legacy", "rich cultural tapestry", "vibrant community", "dynamic hub of activity and culture", "scenic", "lively atmosphere"

**Fix**: Be precise. Instead of "stunning natural beauty," name the specific feature — a granite cliff face, a mangrove estuary, whatever it actually is.

### 3. Editorializing & Reader-Directing

AI inserts meta-commentary, telling the reader what to pay attention to.

**Flagged phrases**: "It's important to note/remember/consider", "It is worth mentioning", "No discussion would be complete without", "In this article, we will explore...", "Let's dive in / Let's explore", "As we can see...", "This brings us to..."

**Fix**: Delete the meta-commentary. Just present the information.

### 4. Tailing "-ing" Clauses (Superficial Analysis)

AI appends present participle phrases to inject fake significance.

**Pattern**: Sentence ending with ", [verb]-ing [abstract claim]"

**Common verbs**: highlighting, emphasizing, reflecting, ensuring, underscoring, demonstrating, showcasing, reinforcing, fostering, paving the way for

**Example**:
> "Consumers benefit from the flexibility to use their preferred mobile wallet at participating merchants, improving convenience."

**Fix**: If the -ing clause adds real information, make it its own sentence. If it restates the main clause, delete it.

### 5. Negative Parallelism ("Not X — It's Y")

One of the most distinctive AI patterns. Appears with unusually high frequency, sometimes multiple times per paragraph.

**Flagged patterns**: "It's not just X — it's Y", "It's not merely X; it's Y", "More than just X, it's Y"

**Example**:
> "It's not just about the beat riding under the vocals; it's part of the aggression and atmosphere."

**Fix**: State the point directly. "The beat contributes to the track's aggression and atmosphere."

### 6. Conjunctive Adverb Overuse

AI uses a narrow set of transitions at high frequency.

**Flagged when overused**: "Moreover", "Furthermore", "Additionally", "However" (at sentence start), "In addition", "In contrast", "On the other hand", "Notably", "Consequently", "Nevertheless"

**Fix**: Delete the transition and see if the sentence still flows. If it does, you didn't need it. If you do need one, pick something that fits the specific logical relationship.

### 7. Em Dash Overuse

AI uses em dashes (—) more frequently than human writers, and in places where commas, parentheses, or colons are more natural.

**Fix**: For each em dash, ask: would a comma, parentheses, or colon work here? If yes, use that instead. Reserve em dashes for genuine dramatic interruption.

### 8. Weasel Attribution

AI attributes claims to vague unnamed authorities.

**Flagged phrases**: "Observers have cited...", "Industry reports suggest...", "Some critics argue...", "Experts believe...", "Studies have shown...", "Many consider...", "It is widely regarded as...", "Scholars have noted..."

**Fix**: Either cite a specific source ("According to a 2024 McKinsey report...") or own the claim.

### 9. Formatting Tells

**Bolded-Label Bullets**: Each item has a **Bolded Label:** followed by a sentence restating the label.
> - **Scalability:** The system is designed to scale easily across different use cases.

**Excessive Boldface**: AI bolds key terms far more than humans do.

**Overly Symmetric Structure**: Neat three-part or five-part structures where content doesn't naturally split that way. Real arguments are often asymmetric.

### 10. Uniform Rhythm

AI produces text where every sentence is roughly the same length and every paragraph the same size. This creates a metronomic, lifeless quality.

**Fix**: Consciously vary sentence length. Follow a long complex sentence with a short one. Let paragraph sizes reflect the natural weight of each point.

### 11. Vocabulary Red Flags

Words disproportionately common in AI output:

"delve/delving into", "tapestry" (metaphorical), "multifaceted", "nuanced" (especially "nuanced understanding"), "landscape" (metaphorical: "the evolving landscape of..."), "paradigm/paradigm shift", "holistic", "synergy", "foster/fostering" (especially "fostering an atmosphere of..."), "realm" (metaphorical), "spearhead", "underscore", "navigating" (metaphorical), "leverage" (as verb), "robust", "comprehensive", "cutting-edge", "groundbreaking", "innovative" (without specifying what), "seamless", "empower/empowering"

None of these are banned. The signal is in frequency and co-occurrence. If your paragraph uses "delve", "tapestry", "nuanced", and "landscape" together, that's the problem.

### 12. Opening Line Patterns

**Flagged openers**: "In today's [world/era/landscape]...", "In the ever-evolving world of...", "When it comes to [topic]...", "[Topic] has become increasingly important in recent years...", "As [field] continues to evolve..."

**Fix**: Start with your actual point, a concrete detail, or a question.

### 13. Closing Patterns

**Flagged closers**: "In conclusion, [restatement]", "As we look to the future...", "Only time will tell...", "One thing is clear: [vague optimism]", "By [doing X], we can [achieve Y], ensuring [Z] for generations to come."

**Fix**: End when you're done making your point. A strong final sentence on something specific beats a paragraph of summary.

---

## Quick Self-Check

After writing, ask:
1. Did I claim anything is "significant" or "important" without showing why?
2. Do more than two consecutive sentences start with conjunctive adverbs?
3. Are my paragraphs all roughly the same length?
4. Would deleting any trailing ", -ing..." clause lose actual information?
5. Did I use "not just X, but Y" more than once?
6. Can I find three or more Vocabulary Red Flag words?
7. Does my opening start with "In today's..." or "When it comes to..."?
8. Did I attribute anything to unnamed "experts" or "observers"?

If three or more are yes, revise.

---

## Caveats

- English text only.
- Not every flagged word is a problem. Context matters. Flag **patterns and clusters**, not isolated words.
- The goal is natural, well-written text — not tricking AI detectors.
- Some tells (em dashes, for instance) are also used by skilled human writers. The question is always frequency, placement, and co-occurrence.
