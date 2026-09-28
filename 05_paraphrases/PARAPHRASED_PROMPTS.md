# Exact paraphrased prompts

The robustness experiment used the following three shared Task replacements for all four arms in ten cases: **10 cases × 4 arms × 3 variants = 120 generated outputs**. The cases were `case_012`, `case_032`, `case_056`, `case_070`, `case_071`, `case_077`, `case_078`, `case_081`, `case_084`, and `case_091`.

## Variant 1 (`paraphrase_1`)

```text
Task: Based only on the variables shown, briefly interpret today's status, identify any clear concern, recommend a concrete Day t+1 training/activity adjustment, cite exact input variables, and set a safety flag.
```

## Variant 2 (`paraphrase_2`)

```text
Task: Review the listed data only. Provide a concise current-state interpretation, grounded risk or concern, specific next-day adjustment, exact-variable supporting evidence, and a safety flag.
```

## Variant 3 (`paraphrase_3`)

```text
Task: Using only this prompt's visible data, write the required JSON with current status, risk identification, Day t+1 adjustment, exact evidence variables, and safety flag.
```

## How each complete prompt was formed

Start with the corresponding arm's original prompt in [FOUR_ARM_PROMPTS.md](../01_prompts/FOUR_ARM_PROMPTS.md), using the same case-specific input values. Replace the entire first span beginning with `Task: ` and ending immediately before the newline preceding `Grounding rules:` with one of the exact texts above. The system prompt and all user-prompt text outside this span remain unchanged, including the displayed input data, grounding rules, safety-flag rules, JSON instructions, and Arm D trend requirements.

These are three shared replacement texts, not 120 independently written paraphrases. The same replacement was applied to Arms A, B, C, and D.

## Scope of the manipulation

The replacement removes the original arm-specific Task wording. In particular, Arm B's sentence `Use generic monitoring language; do not add specialized sport-science concepts beyond the listed data.` was inside the replaced span and is absent from all three variants. These are the prompts actually used; retaining other prompt sections does not establish strict semantic equivalence of every original Task constraint.

Only the three code blocks above are replacement prompt content. The surrounding text explains their historical use. Fully instantiated prompts additionally require the case-specific values and, for Arm D, the saved availability-list string.
