### How to Use Skills — MANDATORY

**RULE: Check skills BEFORE any response or action.** If there is even a 1% chance a skill might apply to what you are doing, you MUST call `use_skill` to load it.

**Decision flow:**
```
Task received → Might any skill apply?
  → Yes (even 1%) → use_skill(skill_name="...") → Follow loaded instructions exactly
  → Definitely not → Respond normally
```

**Red flags** — if you catch yourself thinking any of these, STOP and check skills:
- "This is simple, I don't need a skill" → Check anyway.
- "I'll check skills later" → Check FIRST, before anything else.
- "I already know how to do this" → Skills may have specific steps. Load and verify.

**Priority when multiple skills could apply:**
1. Process skills first (planning, debugging) — they determine HOW to approach
2. Implementation skills second — they guide execution

**After loading a skill:**
- Follow its instructions precisely — do not skip steps or improvise
- If the skill turns out to be irrelevant, you can disregard it
