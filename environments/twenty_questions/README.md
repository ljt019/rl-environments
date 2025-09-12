# Twenty Questions

> ⚠️ **Work in Progress**: This environment is currently under active development and may have breaking changes or unexpected behavior.

### Overview
- **Environment ID**: `twenty-questions`
- **Short description**: Multi-turn game where models try to guess a secret word/object by asking strategic yes/no questions within 20 turns.
- **Tags**: game, multi-turn
- **Socials**: [Github @ljt019](https://github.com/ljt019), [Hf @ljt019](https://huggingface.co/ljt019), [X @Ljt019117161](https://x.com/Ljt019117161)

### Datasets
- **Primary dataset(s)**: 
  - [ljt019/twenty-questions-600](https://huggingface.co/datasets/ljt019/twenty-questions-600): 600 diverse objects/concepts for guessing

### Task & Scoring
- **Type**: multi-turn reasoning game
- **Parser**: XMLParser with fields `["think", ("question", "guess")]` and `answer_field="guess"` to extract final guesses
- **Rubric overview**: Weighted scoring based on victory, efficiency, and format adherence

**Game Mechanics:**

This environment uses an LLM to generate env responses:
**Answerer LLM** (environment): Knows the answer, provides consistent yes/no answers

Models receive:
1. System prompt explaining the twenty questions game rules and format requirements
2. Previous conversation history
3. Responses from the answerer LLM ("Yes", "No", or "I don't know")
4. Turn limit information (up to 20 questions/guesses)

Models must respond with either:
- `<question>Is it alive?</question>` for yes/no questions
- `<guess>elephant</guess>` for final answer attempts

**Expected Response Format:**
```
<think>
Based on the answers so far, it seems to be a living creature that's large...
</think>

<question>Does it live in Africa?</question>
```

Or when ready to guess:
```
<think>
All the clues point to this being an elephant.
</think>

<guess>elephant</guess>
```

The game continues until:
- **Victory**: Model correctly guesses the answer word/object
- **Turn Limit**: 20 questions/guesses are used up
- **Wrong Guess**: Model makes an incorrect final guess

### Quickstart

Run an evaluation (requires answerer model configuration):
```bash
uv run vf-eval twenty-questions -a '{"answerer_model": "gpt-4o-mini", "answerer_base_url": "https://api.openai.com/v1"}'
```

Or with explicit API key:
```bash
uv run vf-eval twenty-questions -a '{"answerer_model": "gpt-4o-mini", "answerer_base_url": "https://api.openai.com/v1", "answerer_api_key": "your-api-key"}'
```

Browse results
```bash
uv run vf-tui
```

## Environment Arguments

| Arg                  | Type | Default      | Description                                    |
| -------------------- | ---- | ------------ | ---------------------------------------------- |
| `answerer_model`     | str  | **Required** | Model name for answerer LLM (knows the answer) |
| `answerer_base_url`  | str  | **Required** | Base URL for answerer LLM API                 |
| `answerer_api_key`   | str  | `OPENAI_API_KEY` env var | API key for answerer LLM |

**Environment Variables:**
- `OPENAI_API_KEY`: API key for the answerer LLM (used when `answerer_api_key` is not provided)

---

## Metrics

| Metric                           | Weight | Meaning                                               |
| -------------------------------- | ------ | ----------------------------------------------------- |
| `reward`                         | -      | Final weighted rubric score (0.0 to 1.8)             |
| `victory_reward`                 | 1.0    | Full reward (1.0) if answer is correctly guessed     |
| `efficiency_reward`              | 0.5    | Bonus reward for guessing with fewer questions       |
| `format_reward`                  | 0.3    | Reward for proper XML format usage                   |

---
