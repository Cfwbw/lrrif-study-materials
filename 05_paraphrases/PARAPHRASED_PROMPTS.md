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

Start with the corresponding arm's original prompt in [FOUR_ARM_PROMPTS](../01_prompts/FOUR_ARM_PROMPTS), using the same case-specific input values. Replace the entire first span beginning with `Task: ` and ending immediately before the newline preceding `Grounding rules:` with one of the exact texts above. The system prompt and all user-prompt text outside this span remain unchanged, including the displayed input data, grounding rules, safety-flag rules, JSON instructions, and Arm D trend requirements.

## Scope of the manipulation

The three variants were used as semantically equivalent paraphrases of the shared response task: interpret the current state using only visible data, identify grounded concerns, suggest a Day t+1 adjustment, cite exact input variables, and assign a safety flag in the required JSON. Here, semantic equivalence refers to these shared response requirements within the retained prompt context; the Task wording varies across the three stored variants.

Each variant replaced the original arm-specific Task wording as described above. The system prompt, visible case data, grounding rules, safety-flag rules, output schema, and Arm D trend requirements remained unchanged. The exact variant texts and replacement span above define the implemented manipulation.
