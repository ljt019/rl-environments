import random
from typing import Optional, Tuple

from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State

from .emulator import (
    BOARD_SIZE,
    SHIPS,
    BattleshipEmulator,
    Coordinate,
    GameState,
    MoveOutcome,
)

BATTLESHIP_SYSTEM_PROMPT = """
You are playing Battleship.

After every turn, the environment sends ONE user message containing the current game state in the following format:

<result move="c3" value="hit|miss|sunk|invalid|victory"/>
<remaining carrier="N" battleship="N" cruiser="N" submarine="N" destroyer="N"/>
<board>
(? unknown, o miss, x hit, s sunk)
10x10 grid representing current board state
</board>

Make your move in this format:

<guess>[coordinate]</guess>
"""

BATTLESHIP_INITIAL_MESSAGE = """
Goal
 - Sink all enemy ships by guessing coordinates.

Coordinate format
  - Column letters (a-j) + row numbers (1-10), e.g., e5.

Symbols in <board>
  ? unknown   o miss   x hit (unsunk)   s sunk-ship part

Per-turn tags (sent each turn)
  - <result move="c3" value="hit|miss|sunk|invalid|victory"/> outcome of your last shot
  - <remaining carrier="…" …/> ships still afloat
  - <board> header line + 10 rows </board> current board representation

Ship sizes
  Carrier (5) • Battleship (4) • Cruiser (3) • Submarine (3) • Destroyer (2)

Important rules
  - NEVER guess a cell that isn't marked "?" (unknown) on the board.
  - Guessing previously guessed cells (marked o, x, or s) is invalid.

<remaining carrier="1" battleship="1" cruiser="1" submarine="1" destroyer="1" />
<board>
   a b c d e f g h i j
 1 ? ? ? ? ? ? ? ? ? ?
 2 ? ? ? ? ? ? ? ? ? ?
 3 ? ? ? ? ? ? ? ? ? ?
 4 ? ? ? ? ? ? ? ? ? ?
 5 ? ? ? ? ? ? ? ? ? ?
 6 ? ? ? ? ? ? ? ? ? ?
 7 ? ? ? ? ? ? ? ? ? ?
 8 ? ? ? ? ? ? ? ? ? ?
 9 ? ? ? ? ? ? ? ? ? ?
10 ? ? ? ? ? ? ? ? ? ?
</board>

Next move:
"""


def generate_games(num_games: int, seed: int) -> Dataset:
    """Generate a dataset of Battleship games.

    Args:
        num_games: Number of games to generate
        seed: If provided, makes the dataset reproducible. If None, generates random games.

    Returns:
        Dataset containing game configurations that can be split by the user
    """
    rng = random.Random(seed)

    # Generate unique seeds for each game
    game_seeds = [rng.randint(0, 1_000_000) for _ in range(num_games)]

    dataset_list = [
        {
            "question": BATTLESHIP_INITIAL_MESSAGE,
            "info": {
                "seed": game_seeds[i],
            },
        }
        for i in range(num_games)
    ]

    return Dataset.from_list(dataset_list)


def load_environment(
    max_turns: Optional[int] = None, num_games: Optional[int] = None, seed: Optional[int] = None, **kwargs
):
    """Load and configure the Battleship environment with standard game settings."""

    if num_games is None:
        num_games = 1000

    if seed is None:
        seed = 5656

    if max_turns is None:
        max_turns = 50

    dataset = generate_games(num_games, seed).train_test_split(test_size=0.2)

    parser = vf.XMLParser(
        fields=["think", "guess"],
        answer_field="guess",
    )

    def victory_reward(state: State) -> float:
        return 1.0 if state.get("victory", False) else 0.0

    def hit_reward(state: State) -> float:
        emulator_state: GameState = state["emulator_state"]

        total_hits = sum(len(ship.hits) for ship in emulator_state.ships)
        return min(0.5, total_hits * 0.03)  # 0.03 per hit, max 0.5

    def strategic_hit_reward(state: State) -> float:
        """Bonus reward on top of hit reward for following up on hits"""

        emulator_state: GameState = state["emulator_state"]

        all_hits = set()
        for ship in emulator_state.ships:
            all_hits.update(ship.hits)

        # Count adjacent hits (follow-up)
        adjacent_hits = 0
        hits_list = list(all_hits)
        for hit1 in hits_list:
            for hit2 in hits_list:
                if hit1 != hit2:
                    coord1 = Coordinate.from_str(hit1)
                    coord2 = Coordinate.from_str(hit2)
                    if _are_adjacent(coord1, coord2):
                        adjacent_hits += 1
        strategic_pairs = adjacent_hits // 2
        return strategic_pairs * 0.03

    def coverage_efficiency_reward(state: State) -> float:
        emulator_state: GameState = state["emulator_state"]

        guesses = set(emulator_state.history)

        active_hits = set()
        for ship in emulator_state.ships:
            if not ship.is_sunk:
                active_hits.update(ship.hits)

        regions = set()
        region_size = max(1, BOARD_SIZE // 3)
        for guess in guesses:
            if len(guess) >= 2:
                col = ord(guess[0].lower()) - ord("a")
                row = int(guess[1:]) - 1
                region = (col // region_size, row // region_size)
                regions.add(region)

        total_regions = (BOARD_SIZE // region_size) ** 2
        coverage_score = len(regions) / float(total_regions)

        if active_hits:
            coverage_score *= 0.5  # Half reward when following up hits

        return min(0.3, coverage_score * 0.4)  # Max 0.3, but scaled down

    rubric = vf.Rubric(
        funcs=[
            victory_reward,
            hit_reward,
            strategic_hit_reward,
            coverage_efficiency_reward,
            parser.get_format_reward_func(),
        ],
        weights=[0.6, 0.4, 0.5, 0.3, 0.2],
        parser=parser,
    )

    return BattleshipEnv(
        dataset=dataset["train"],
        eval_dataset=dataset["test"],
        system_prompt=BATTLESHIP_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )


class BattleshipEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emulator = BattleshipEmulator()

    def env_response(self, messages: Messages, game_state: State, **kwargs) -> Tuple[Messages, State]:
        """Define how the environment responds."""

        if "emulator_state" not in game_state:
            temp_emulator = BattleshipEmulator(seed=game_state.get("seed", None))
            game_state["emulator_state"] = temp_emulator.get_state()
            game_state["turn"] = 0

        if not messages:
            return [], game_state

        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            return [], game_state

        move_str = self.parser.parse_answer(last_msg["content"])
        if not move_str:
            return [
                {
                    "role": "user",
                    "content": "Please provide a valid move in the format <guess>[coordinate]</guess>",
                }
            ], game_state

        try:
            move = Coordinate.from_str(move_str)
        except ValueError:
            return [
                {
                    "role": "user",
                    "content": "Invalid coordinate format. Use format like 'e5'",
                }
            ], game_state

        current_game_state = game_state["emulator_state"]
        self.emulator.restore_state(current_game_state)

        outcome = self.emulator.step(move)

        final_state = self.emulator.get_state()
        game_state["emulator_state"] = final_state
        game_state["turn"] += 1

        feedback = self._format_game_state(outcome, final_state)

        if final_state.game_over:
            game_state["victory"] = True

        return [{"role": "user", "content": feedback}], game_state

    def _format_game_state(self, outcome: MoveOutcome, game_state: GameState) -> str:
        """Format the current game state as XML tags."""
        if outcome.invalid:
            result_value = "invalid"
        elif game_state.game_over and outcome.hit:
            result_value = "victory"
        elif outcome.sunk:
            result_value = "sunk"
        elif outcome.hit:
            result_value = "hit"
        else:
            result_value = "miss"

        remaining_counts = {ship.name.lower(): 0 for ship in SHIPS}
        for ship in game_state.ships:
            if not ship.is_sunk:
                ship_key = ship.name.lower()
                if ship_key in remaining_counts:
                    remaining_counts[ship_key] = 1

        board = game_state.board

        cols = "abcdefghij"[:BOARD_SIZE]
        grid_lines = [f"   {' '.join(cols)}"]
        for row in range(1, BOARD_SIZE + 1):
            line = f"{row:2d}"
            for col in cols:
                cell = board.cells[f"{col}{row}"]
                line += f" {cell}"
            grid_lines.append(line)

        grid_content = "\n".join(grid_lines)

        # Build response
        result = f'<result move="{outcome.move}" value="{result_value}"/>\n'
        result += f'<remaining carrier="{remaining_counts.get("carrier", 0)}" battleship="{remaining_counts.get("battleship", 0)}" cruiser="{remaining_counts.get("cruiser", 0)}" submarine="{remaining_counts.get("submarine", 0)}" destroyer="{remaining_counts.get("destroyer", 0)}" />\n'
        result += f"<board>\n{grid_content}\n</board>\n"

        if not game_state.game_over:
            result += "\nNext move:"

        return result

    def is_completed(self, messages: Messages, game_state: State, **kwargs) -> bool:
        """Check if the game is completed."""
        return game_state.get("victory", False)


def _are_adjacent(coord1: Coordinate, coord2: Coordinate) -> bool:
    """Check if two coordinates are adjacent (orthogonally)."""
    try:
        col1, row1 = ord(coord1.col) - ord("a"), coord1.row
        col2, row2 = ord(coord2.col) - ord("a"), coord2.row

        return abs(col1 - col2) + abs(row1 - row2) == 1
    except AttributeError:
        return False
