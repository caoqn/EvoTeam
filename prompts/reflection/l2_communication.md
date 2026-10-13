## L2 Reflection — Pairwise Collaboration & Teammate Profiling

You are now in the **L2 Reflection phase**. Reflect on your interactions with other agents during this task.

### Goal
Improve future team collaboration by updating observations about teammates' working styles. Canonical handoff contracts are reflected separately after L2 and saved in `handoff_rules.json`.

**Scope boundary:** A teammate profile is a compact, general capability summary. It must not duplicate a handoff rule. Do not record handoff triggers, `rule_id` values, `from_agent`/`to_agent` routing, payload schemas, required evidence, verification steps, fallback procedures, or instructions about when to transfer information. Those belong exclusively to the team-scoped handoff-rule reflector.

### Discussion Protocol

**Default action: Skip.** If collaboration went smoothly, record useful observations with `update_teammate_profile`, then call `skip_l2_reflection`. **Do NOT send discussion messages just to be polite or ask generic questions.**

**Only initiate a discussion if there was a concrete problem** — e.g., misunderstood requirements, unclear handoffs, wrong outputs that needed rework, or repeated failures. The "Your L2 Discussion Assignments" section at the end tells you your role **if** you need to discuss:

- **If you are the initiator** for a partner: Send them a brief message **about the specific problem**. Then use `wait_for_replies` to collect their response.
- **If you are waiting** for a partner: Do NOT send them a message first. Wait for them to reach out, then reply thoughtfully.

**Important**: Do not send discussion messages to partners you are assigned to wait for. This prevents duplicate crossed conversations.

### ⚠ CRITICAL: Always Reply to Received Messages

**If you receive a message from another agent (via `wait_for_replies` or `read_messages`), you MUST reply to them using `send_message` before doing anything else.** The other agent is waiting for your reply — if you don't send one, they will be stuck indefinitely.

- `update_teammate_profile` is an internal recording tool — it does NOT send any message to the other agent.
- Only `send_message` actually delivers a reply. Always call `send_message(to=<sender>)` first, then do your internal recording.

### After Each Discussion

For each partner (whether you initiated or responded), use **`update_teammate_profile(teammate_name, profile)`** to record observations in YAML:
   ```yaml
   reliability: high | medium | low
   strengths:
     - "general stable pattern"
   weaknesses:
     - "general stable limitation"
   communication_style: "one sentence"
   notes:
     - "brief general observation"
   # Optional: delete only clearly one-off, contradicted, or obsolete items.
   # List the exact old text under its field; omission does not delete it.
   remove:
     strengths: []
     weaknesses: []
     notes: []
   ```

   **Profile quality rules:**
   - Submit a **complete curated current profile** for the teammate, not just newly observed items
   - Preserve useful stable patterns from the existing profile, and consolidate similar observations into one clearer statement
   - By default, do not delete uncertain information. To delete an item, include its exact old text in the optional `remove` mapping
   - If a concrete event can be generalized into a stable capability or limitation, write the generalized statement and explicitly remove the concrete old item
   - Keep only stable capability, reliability, limitation, and communication observations; do not preserve task-specific evidence or handoff procedures

### Complete L2

When you have finished recording observations for all the partners **you** need to initiate with, call `skip_l2_reflection(reason)` to **mark your L2 updates as complete**.

**After calling `skip_l2_reflection`**, you will enter idle. But you **remain responsive**: if another agent reaches out to discuss collaboration with you, you will be woken up. **You MUST reply using `send_message`**, then use `update_teammate_profile` to record useful observations.

The reflection session ends when ALL agents have called `skip_l2_reflection`.

### Rules
- **Skip by default** — Discuss only if there was a real problem.
- **Be concise** — If you do discuss, aim to complete L2 in 2-3 tool calls total. One message, one reply, done.
- **Only profile agents you directly interacted with** during this task
- Be **objective and evidence-based** — use task events as internal evidence, but write only the resulting general pattern
- **Curate when updating** — retain useful existing patterns, consolidate similar items, and submit the resulting full profile
- Existing list items are retained by default. Use the optional `remove` mapping only for clearly one-off, contradicted, or obsolete observations
- Be **constructive** — the goal is to improve collaboration, not to criticize
- **Do NOT initiate discussion with partners you are assigned to wait for**

### CRITICAL
**You MUST call `skip_l2_reflection`** when your own L2 updates are done. After that, stay responsive — reply if other agents send you L2 discussion messages.
