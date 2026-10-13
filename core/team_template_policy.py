from __future__ import annotations


TEAM_TEMPLATE_POLICY = {
    "team_template_unit": "a reusable multi-agent collaboration pattern",
    "identity_dimensions": [
        "work type",
        "change or investigation scope",
        "required specialist roles",
        "verification strategy",
    ],
    "positive_rules": [
        "Reuse a team template when these collaboration dimensions are substantially the same.",
        "Create a team template only when the task requires a materially different collaboration pattern.",
        "Name and describe a team template in terms of its collaboration pattern.",
    ],
    "negative_rules": [
        "A shared benchmark wrapper alone is not evidence that two tasks share one team template.",
        "Benchmark names, repository names, domains, business topics, entities, and filenames do not define team-template identity.",
        "A different subject or requested feature alone is not evidence for a new team template.",
    ],
}

