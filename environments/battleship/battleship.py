import re
from typing import Tuple

import verifiers as vf
from battleship_emulator import BattleshipEmulator
from verifiers.types import Messages, State

BATTLESHIP_SYSTEM_PROMPT = """
You are playing Battleship.

After every turn, the environment sends ONE user message containing the current game state in tagged form:

<result move="c3" value="hit|miss|sunk|invalid|victory"/>
<remaining carrier="N" battleship="N" cruiser="N" submarine="N" destroyer="N"/>
<state hits="a5 e4" misses="b1 d6" sunk="d5 e5" unknown="83"/>
<grid>
(? unknown, o miss, x hit, s sunk)
10x10 grid representing current board state
</grid>

Rules for you:
1. Finish ships by guessing cells directly adjacent (up, down, left, right—no diagonals) to confirmed hits before exploring new areas.
2. Respond EXACTLY in the following format and nothing else:

{Concise reasoning about the next best shot}
<guess>[coordinate]</guess>
"""

BATTLESHIP_INITIAL_MESSAGE = """
Goal
 - Sink all enemy ships by guessing coordinates.

Coordinate format
  - Column letters (a-j) + row numbers (1-10), e.g., e5.

Symbols in <grid>
  ? unknown   o miss   x hit (unsunk)   s sunk-ship part

Per-turn tags (sent each turn)
  - <result move="c3" value="hit|miss|sunk|invalid|victory"/> outcome of your last shot
  - <remaining carrier="…" …/> ships still afloat
  - <state hits="…" misses="…" sunk="…" unknown="N"/> status of guessed cells
  - <grid> header line + 10 rows </grid> current board representation

Ship sizes
  Carrier (5) • Battleship (4) • Cruiser (3) • Submarine (3) • Destroyer (2)

Important rules
  - NEVER guess a cell that isn't marked "?" (unknown) on the grid.
  - Guessing previously guessed cells (marked o, x, or s) is invalid.

<result move="" value="start"/>
<remaining carrier="1" battleship="1" cruiser="1" submarine="1" destroyer="1" />
<state hits="" misses="" sunk="" unknown="100"/>
<grid>
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
</grid>

Next move:
"""


def load_environment(**kwargs):
    """Load and configure the Battleship environment."""
    from datasets import Dataset

    # Create a simple dataset for battleship games
    num_games = kwargs.get("num_games", 100)
    dataset_list = [
        {
            "question": BATTLESHIP_INITIAL_MESSAGE,
            "seed": i,
            "info": {
                "seed": i,  # removed anyway but yells at me if it's empty
            },
            "answer": "victory",
        }
        for i in range(num_games)
    ]
    dataset = Dataset.from_list(dataset_list)

    # Use built-in XMLParser to extract guess coordinates
    parser = vf.XMLParser(
        fields=["think", "guess"],
        answer_field="guess",
    )

    # Define reward functions
    def victory_reward(completion, answer, **kwargs):
        """Give reward when player achieves victory."""
        try:
            state = kwargs.get("state", {})
            return 1.0 if state.get("victory", False) else 0.0
        except Exception as e:
            print(f"Victory reward error: {e}")
            return 0.0

    def hit_reward(completion, answer, **kwargs):
        """Small reward for each hit, even without winning."""
        try:
            # Extract hits from completion messages
            hits = set()
            if isinstance(completion, list):
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        # Look for hit results in user messages
                        if 'value="hit"' in content or 'value="sunk"' in content:
                            # Extract coordinate from move="coordinate"
                            match = re.search(r'move="([a-j][0-9]+)"', content)
                            if match:
                                hits.add(match.group(1))

            hit_count = len(hits)
            return min(0.5, hit_count * 0.03)  # 0.03 per hit, max 0.5
        except Exception as e:
            print(f"Hit reward error: {e}")
            return 0.0

    def strategic_hit_reward(completion, answer, **kwargs):
        """Bigger reward for follow-up hits showing strategic thinking."""
        try:
            # Extract hits from completion messages
            hits = set()
            if isinstance(completion, list):
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        # Look for hit results in user messages
                        if 'value="hit"' in content or 'value="sunk"' in content:
                            # Extract coordinate from move="coordinate"
                            match = re.search(r'move="([a-j][0-9]+)"', content)
                            if match:
                                hits.add(match.group(1))

            # Count adjacent hits (shows strategic follow-up)
            adjacent_hits = 0
            hits_list = list(hits)
            for hit1 in hits_list:
                for hit2 in hits_list:
                    if hit1 != hit2 and _are_adjacent(hit1, hit2):
                        adjacent_hits += 1
            strategic_pairs = adjacent_hits // 2
            return min(0.3, strategic_pairs * 0.1)  # 0.1 per strategic pair, max 0.3
        except Exception as e:
            print(f"Strategic hit reward error: {e}")
            return 0.0

    def coverage_efficiency_reward(completion, answer, **kwargs):
        """Reward good board coverage, but not when actively hunting ships."""
        try:
            # Extract all guesses and current hits
            guesses = set()
            active_hits = set()  # Hits that aren't part of sunk ships yet

            if isinstance(completion, list):
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")

                        # Extract the guess coordinate
                        move_match = re.search(r'move="([a-j][0-9]+)"', content)
                        if move_match:
                            coord = move_match.group(1)
                            guesses.add(coord)

                            # Track active hits (hits not yet sunk)
                            if 'value="hit"' in content:
                                active_hits.add(coord)
                            elif 'value="sunk"' in content:
                                # When something sinks, remove related hits from active
                                active_hits.discard(coord)

            # Divide board into 3x3 regions for coverage analysis
            regions = set()
            for guess in guesses:
                if len(guess) >= 2:
                    col = ord(guess[0].lower()) - ord("a")  # 0-9
                    row = int(guess[1:]) - 1  # 0-9
                    region = (col // 3, row // 3)  # Creates 9 regions
                    regions.add(region)

            # Base coverage score (0.0 to 1.0)
            coverage_score = len(regions) / 9.0  # 9 total regions

            # Reduce coverage importance when actively hunting
            if active_hits:
                coverage_score *= 0.5  # Half reward when following up hits

            return min(0.3, coverage_score * 0.4)  # Max 0.3, but scaled down

        except Exception as e:
            print(f"Coverage efficiency reward error: {e}")
            return 0.0

    def tool_call_penalty(completion, answer, **kwargs):
        """Penalize usage of <tool_call> tags in responses."""
        try:
            # Check if completion contains tool_call tags (case insensitive)
            completion_text = str(completion).lower()

            # Count occurrences of tool_call patterns
            tool_call_patterns = [
                "<tool_call>",
                "</tool_call>",
            ]
            for pattern in tool_call_patterns:
                if pattern in completion_text:
                    return -0.2  # Penalty for tool calls
            return 0.0
        except Exception as e:
            print(f"Tool call penalty error: {e}")
            return 0.0

    rubric = vf.Rubric(
        funcs=[
            victory_reward,
            hit_reward,
            strategic_hit_reward,
            coverage_efficiency_reward,
            tool_call_penalty,
            parser.get_format_reward_func(),
        ],
        weights=[0.6, 0.4, 0.5, 0.3, 0.2, 0.2],
    )

    # Return configured environment
    return BattleshipEnv(
        dataset=dataset,
        system_prompt=BATTLESHIP_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )


class BattleshipEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.board_size = kwargs.get("board_size", 10)
        self.ship_sizes = kwargs.get("ship_sizes", [5, 4, 3, 3, 2])
        self.ship_names = ["carrier", "battleship", "cruiser", "submarine", "destroyer"]

    def env_response(self, messages: Messages, game_state: State, **kwargs) -> Tuple[Messages, State]:
        """Define how the environment responds."""
        # Initialize game state if needed
        if "emulator_state" not in game_state:
            emulator = BattleshipEmulator(
                board_size=self.board_size,
                ship_sizes=self.ship_sizes,
                seed=game_state.get("seed", None),
            )
            game_state["emulator_state"] = emulator.get_state().to_dict()
            game_state["turn"] = 0
            game_state["initialized"] = False

        # Get the last message from the assistant
        if not messages:
            return [], game_state

        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            return [], game_state  # No response if not assistant message

        # Extract move from assistant message
        move = self._extract_move(last_msg["content"])
        if not move:
            return [
                {
                    "role": "user",
                    "content": "Please provide a valid move in the format <guess>[coordinate]</guess>",
                }
            ], game_state

        # Process the move
        emulator = self._restore_emulator(game_state["emulator_state"])
        observation, hit, sunk, game_over, invalid = emulator.step(move)

        # Update state
        game_state["emulator_state"] = emulator.get_state().to_dict()
        game_state["turn"] += 1

        # Generate feedback
        feedback = self._format_game_state(move, hit, sunk, game_over, invalid, emulator.get_state())

        if game_over:
            game_state["victory"] = True

        return [{"role": "user", "content": feedback}], game_state

    def _restore_emulator(self, emulator_dict):
        """Restore BattleshipEmulator from dictionary state."""
        emulator = BattleshipEmulator(
            board_size=emulator_dict["board"]["board_size"],
            ship_sizes=[ship["size"] for ship in emulator_dict["ships"]],
        )

        # Restore state
        emulator.turn_count = emulator_dict["turn_count"]
        emulator.game_over = emulator_dict["game_over"]
        emulator.last_move_invalid = emulator_dict["last_move_invalid"]
        emulator.history = emulator_dict["history"].copy()
        emulator.board = emulator_dict["board"]["cells"].copy()

        # Restore ships
        emulator.ships = []
        for ship_dict in emulator_dict["ships"]:
            from battleship_emulator.models import ShipStatus

            ship = ShipStatus(
                name=ship_dict["name"],
                size=ship_dict["size"],
                coords=ship_dict["coords"],
                hits=set(ship_dict["hits"]),
            )
            emulator.ships.append(ship)

        return emulator

    def _extract_move(self, content: str) -> str:
        """Extract coordinate from assistant message using XMLParser."""
        # Create a temporary parser instance to extract the guess
        temp_parser = vf.XMLParser(fields=["guess"], answer_field="guess")
        return temp_parser.parse_answer(content) or ""

    def _format_game_state(
        self,
        move: str,
        hit: bool,
        sunk: bool,
        game_over: bool,
        invalid: bool,
        game_state,
    ) -> str:
        """Format the current game state as XML tags."""
        # Determine result value
        if invalid:
            result_value = "invalid"
        elif game_over and hit:
            result_value = "victory"
        elif sunk:
            result_value = "sunk"
        elif hit:
            result_value = "hit"
        else:
            result_value = "miss"

        # Count remaining ships
        remaining_counts = {name: 0 for name in self.ship_names}
        for ship in game_state.ships:
            if not ship.is_sunk:
                # Map ship names to standard names
                ship_name_lower = ship.name.lower()
                if "carrier" in ship_name_lower or ship.size == 5:
                    remaining_counts["carrier"] = 1
                elif "battleship" in ship_name_lower or ship.size == 4:
                    remaining_counts["battleship"] = 1
                elif "cruiser" in ship_name_lower or (ship.size == 3 and remaining_counts["cruiser"] == 0):
                    remaining_counts["cruiser"] = 1
                elif "submarine" in ship_name_lower or (ship.size == 3 and remaining_counts["submarine"] == 0):
                    remaining_counts["submarine"] = 1
                elif "destroyer" in ship_name_lower or ship.size == 2:
                    remaining_counts["destroyer"] = 1

        # Get board state
        board = game_state.board
        hits = []
        misses = []
        sunk_cells = []

        for pos, val in board.cells.items():
            if val == "x":
                hits.append(pos)
            elif val == "o":
                misses.append(pos)
            elif val == "s":
                sunk_cells.append(pos)

        unknown_count = len(board.unknown_cells)

        # Format grid with proper alignment
        grid_lines = ["   a b c d e f g h i j"]  # Header with 3 spaces before 'a'
        for row in range(1, 11):
            line = f"{row:2d}"  # Right-align row numbers in 2-character field
            for col in "abcdefghij":
                cell = board.cells[f"{col}{row}"]
                line += f" {cell}"
            grid_lines.append(line)

        grid_content = "\n".join(grid_lines)

        # Build response
        result = f'<result move="{move}" value="{result_value}"/>\n'
        result += f'<remaining carrier="{remaining_counts["carrier"]}" battleship="{remaining_counts["battleship"]}" cruiser="{remaining_counts["cruiser"]}" submarine="{remaining_counts["submarine"]}" destroyer="{remaining_counts["destroyer"]}" />\n'
        result += f'<state hits="{" ".join(sorted(hits))}" misses="{" ".join(sorted(misses))}" sunk="{" ".join(sorted(sunk_cells))}" unknown="{unknown_count}"/>\n'
        result += f"<grid>\n{grid_content}\n</grid>\n"

        if not game_over:
            result += "\nNext move:"

        return result

    def is_completed(self, messages: Messages, game_state: State, **kwargs) -> bool:
        """Check if the game is completed."""
        return game_state.get("victory", False)


def _are_adjacent(coord1, coord2):
    """Check if two coordinates are adjacent (orthogonally)."""
    try:
        # Parse coordinates like "a5", "c3"
        col1, row1 = ord(coord1[0].lower()) - ord("a"), int(coord1[1:])
        col2, row2 = ord(coord2[0].lower()) - ord("a"), int(coord2[1:])

        # Adjacent if exactly 1 step away horizontally or vertically
        return abs(col1 - col2) + abs(row1 - row2) == 1
    except (ValueError, IndexError):
        return False
