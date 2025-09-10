# Battleship

### Overview
- **Environment ID**: `battleship`
- **Short description**: Multi-turn environment where models play the classic game of Battleship by strategically guessing coordinates to locate and sink all enemy ships.
- **Tags**: game, multi-turn, strategy
- **Source Implementation**: Original implementation for RL training
- **Socials**: [Github @ljt019](https://github.com/ljt019), [Hf @ljt019](https://huggingface.co/ljt019), [X @Ljt019117161](https://x.com/Ljt019117161)

### Datasets
- **Primary dataset(s)**: 
  - Generated gameplay scenarios (configurable number of games)

### Task & Scoring
- **Type**: multi-turn strategy game
- **Parser**: XMLParser extracts coordinates from `<guess>COORDINATE</guess>` tags
- **Rubric overview**: Weighted scoring based on victory, hits, strategic play, and board coverage

**Game Mechanics:**

Models receive:
1. Current game state with XML tags showing results, remaining ships, hit/miss/sunk status
2. Visual grid board representation (10x10 board)
3. Ship fleet: Carrier(5), Battleship(4), Cruiser(3), Submarine(3), Destroyer(2)

Models must respond with: `<guess>COORDINATE</guess>` using format like `a1`, `e5`, `j10`.

**Grid Symbols:**
- `?` unknown/unguessed cell
- `o` miss
- `x` hit (ship not sunk)  
- `s` sunk ship part

**Expected Response Format:**
```
Strategic reasoning about next move
<guess>e5</guess>
```

The game continues until:
- **Victory**: All ships are sunk
- **Turn Limit**: Maximum turns reached

### Quickstart

Run an evaluation with default settings:
```bash
uv run vf-eval battleship
```

Browse results
```bash
uv run vf-tui
```

## Environment Arguments

| Arg           | Type         | Default           | Description                              |
| ------------- | ------------ | ----------------- | ---------------------------------------- |
| `max_turns`   | int          | `50`              | Maximum number of moves allowed          |
| `num_games`   | int          | `1000`            | Number of games in dataset               |
| `seed`        | int          | `5656`            | Seed for reproducible games      |

---

## Metrics

| Metric                           | Weight | Meaning                                               |
| -------------------------------- | ------ | ----------------------------------------------------- |
| `reward`                         | -      | Final weighted rubric score (0.0 to 2.01)             |
| `victory_reward`                 | 0.6    | Full reward (1.0) if all ships are sunk              |
| `hit_reward`                     | 0.4    | Reward for each hit (0.03 per hit)          |
| `strategic_hit_reward`           | 0.5    | Reward for adjacent hits to reward follow up   |
| `coverage_efficiency_reward`     | 0.3    | Reward for good board coverage and exploration        |
| `format_reward`                  | 0.2    | Reward for proper `<guess>COORDINATE</guess>` format  |

---