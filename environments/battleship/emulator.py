import os
import random
import string
import sys

sys.path.append(os.path.dirname(__file__))
from models import BoardState, GameState, ShipStatus

# Default ship configuration: carrier(5), battleship(4), cruiser(3), submarine(3), destroyer(2)
DEFAULT_SHIP_SIZES = [5, 4, 3, 3, 2]
DEFAULT_SHIP_NAMES = ["Carrier", "Battleship", "Cruiser", "Submarine", "Destroyer"]


class BattleshipEmulator:
    def __init__(self, board_size=10, ship_sizes=None, seed=None):
        # Board configuration
        self.board_size = board_size
        self.cols = string.ascii_lowercase[:board_size]
        self.rows = [str(r) for r in range(1, board_size + 1)]

        # Ship configuration
        self.ship_sizes = ship_sizes if ship_sizes else DEFAULT_SHIP_SIZES

        # Validate ship sizes are feasible for the board
        if any(size > self.board_size for size in self.ship_sizes):
            raise ValueError("All ship sizes must be less than or equal to the board size")

        # Instance-level RNG for reproducibility
        self.rng = random.Random(seed)

        self.reset()

    def reset(self, seed=None):
        # Optionally reseed for deterministic setup
        if seed is not None:
            self.rng.seed(seed)

        # Initialize board state
        self.board = {f"{c}{r}": "?" for c in self.cols for r in self.rows}
        self.ships = []

        # Initialize game state
        self.history = []
        self.turn_count = 0
        self.game_over = False
        self.last_move_invalid = False

        # Place ships on the board
        self._place_ships()

    def _generate_ship_names(self):
        """Generate names for ships based on their sizes."""
        if self.ship_sizes == DEFAULT_SHIP_SIZES:
            return DEFAULT_SHIP_NAMES.copy()

        names_sequence = []
        seen_by_size = {}
        for sz in self.ship_sizes:
            seen_by_size[sz] = seen_by_size.get(sz, 0) + 1
            names_sequence.append(f"size-{sz} #{seen_by_size[sz]}")
        return names_sequence

    def _generate_ship_coordinates(self, size, is_horizontal):
        """Generate coordinates for a ship of given size and orientation."""
        if is_horizontal:
            row = self.rng.choice(self.rows)
            start_idx = self.rng.randint(0, self.board_size - size)
            return [f"{self.cols[start_idx + i]}{row}" for i in range(size)]
        else:
            col = self.rng.choice(self.cols)
            start_idx = self.rng.randint(0, self.board_size - size)
            return [f"{col}{self.rows[start_idx + i]}" for i in range(size)]

    def _place_ships(self):
        """Place all ships on the board without overlapping."""
        names_sequence = self._generate_ship_names()
        occupied = set()

        for idx, size in enumerate(self.ship_sizes):
            placed = False
            while not placed:
                is_horizontal = self.rng.choice([True, False])
                coords = self._generate_ship_coordinates(size, is_horizontal)

                # Check for overlap
                has_overlap = any(c in occupied for c in coords)
                if not has_overlap:
                    self.ships.append(
                        ShipStatus(
                            name=names_sequence[idx],
                            size=size,
                            coords=coords,
                            hits=set(),
                        )
                    )
                    occupied.update(coords)
                    placed = True

    def render(self) -> str:
        """
        Render the current board state as a string.
        """
        # Create header with column letters
        result = "  " + " ".join(self.cols) + "\n"

        # Add each row
        for row in self.rows:
            result += f"{row:>2}"
            for col in self.cols:
                cell = self.board[f"{col}{row}"]
                result += f" {cell}"
            result += "\n"

        return result

    def step(self, move):
        """
        move: string like 'b6'
        returns: observation (rendered string), hit (bool), sunk (bool), game_over (bool), invalid_move (bool)
        """
        move = move.lower().strip()

        # Guard clause for invalid moves
        is_invalid_move = move not in self.board or self.board[move] != "?"
        if is_invalid_move:
            self.last_move_invalid = True
            self.history.append(move)
            self.turn_count += 1
            return self.render(), False, False, self.game_over, True

        # Process valid move
        self.last_move_invalid = False

        # Check if move hits any ship
        hit_ship = None
        for ship in self.ships:
            if move in ship.coords:
                ship.hits.add(move)
                hit_ship = ship
                break

        # Determine move outcome
        hit = hit_ship is not None
        sunk = hit and hit_ship.is_sunk

        # Update board based on outcome
        if hit:
            if sunk:
                # Mark all positions of the sunk ship as 's'
                for coord in hit_ship.coords:
                    self.board[coord] = "s"
            else:
                self.board[move] = "x"
        else:
            self.board[move] = "o"

        # Update game state
        self.history.append(move)
        self.turn_count += 1

        # Check for game over
        all_ships_sunk = all(ship.is_sunk for ship in self.ships)
        if all_ships_sunk:
            self.game_over = True

        return self.render(), hit, sunk, self.game_over, False

    def get_valid_moves(self):
        """
        Returns list of available (unknown) cells.
        """
        unknown_cells = [pos for pos, val in self.board.items() if val == "?"]
        return unknown_cells

    def get_state(self) -> GameState:
        """
        Returns the current game state with full information.

        Returns:
            GameState object with complete game information.
        """
        board_state = BoardState(board_size=self.board_size, cells=self.board.copy())

        return GameState(
            turn_count=self.turn_count,
            game_over=self.game_over,
            last_move_invalid=self.last_move_invalid,
            history=self.history.copy(),
            board=board_state,
            ships=[
                ShipStatus(
                    name=ship.name,
                    size=ship.size,
                    coords=list(ship.coords),
                    hits=set(ship.hits),
                )
                for ship in self.ships
            ],
        )
