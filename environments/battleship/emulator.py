import random
import string
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class Coordinate:
    """Represents a board coordinate like 'b6'."""

    col: str
    row: int

    @classmethod
    def from_str(cls, s: str) -> "Coordinate":
        """Parse coordinate string like 'b6' into Coordinate.

        Args:
            s: Coordinate string (case-insensitive, whitespace is trimmed).
               Examples: 'b6', 'B6', ' a10 '

        Returns:
            Coordinate object with normalized column (lowercase) and row.

        Raises:
            ValueError: If coordinate format is invalid.
        """
        normalized_input = s.lower().strip()
        if len(normalized_input) < 2:
            raise ValueError(f"Invalid coordinate format: {s}")
        col = normalized_input[0]
        row = int(normalized_input[1:])
        return cls(col=col, row=row)

    def to_str(self) -> str:
        """Convert coordinate back to string format like 'b6'."""
        return f"{self.col}{self.row}"


@dataclass
class ShipPlacement:
    """Represents a ship's initial placement."""

    name: str
    size: int
    coords: List[str]


@dataclass
class ShipStatus:
    """Represents the current status of a ship during gameplay."""

    name: str
    size: int
    coords: List[str]
    hits: Set[str]

    @property
    def is_sunk(self) -> bool:
        """True if all coordinates have been hit."""
        coords_set = set(self.coords)
        hits_set = set(self.hits)
        return hits_set == coords_set

    @property
    def remaining_cells(self) -> int:
        """Number of unhit cells remaining."""
        total_cells = len(self.coords)
        hit_cells = len(self.hits)
        return total_cells - hit_cells


@dataclass
class BoardState:
    """Represents the current state of the game board."""

    board_size: int
    cells: Dict[str, str]  # "?" unknown, "x" hit, "o" miss, "s" sunk

    @property
    def unknown_cells(self) -> List[str]:
        """List of cells that haven't been guessed yet."""
        return [pos for pos, val in self.cells.items() if val == "?"]


@dataclass
class MoveOutcome:
    """Represents the result of a single move."""

    move: str
    hit: bool
    sunk: bool
    invalid: bool
    render: str


@dataclass
class GameState:
    """Complete game state snapshot."""

    turn_count: int
    game_over: bool
    last_move_invalid: bool
    history: List[str]
    board: BoardState
    ships: List[ShipStatus]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "turn_count": self.turn_count,
            "game_over": self.game_over,
            "last_move_invalid": self.last_move_invalid,
            "history": self.history,
            "board": {
                "board_size": self.board.board_size,
                "cells": self.board.cells,
                "unknown_cells": self.board.unknown_cells,
            },
            "ships": [
                {
                    "name": ship.name,
                    "size": ship.size,
                    "coords": ship.coords,
                    "hits": list(ship.hits),
                    "is_sunk": ship.is_sunk,
                    "remaining_cells": ship.remaining_cells,
                }
                for ship in self.ships
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GameState":
        """Create GameState from dictionary representation."""
        # Reconstruct BoardState
        board_data = data["board"]
        board = BoardState(board_size=board_data["board_size"], cells=board_data["cells"])

        # Reconstruct ships
        ships = []
        for ship_data in data["ships"]:
            ship = ShipStatus(
                name=ship_data["name"], size=ship_data["size"], coords=ship_data["coords"], hits=set(ship_data["hits"])
            )
            ships.append(ship)

        return cls(
            turn_count=data["turn_count"],
            game_over=data["game_over"],
            last_move_invalid=data["last_move_invalid"],
            history=data["history"],
            board=board,
            ships=ships,
        )


# Standard Battleship game constants - 10x10 board with fixed ship configuration
BOARD_SIZE = 10


@dataclass
class ShipType:
    """Standard Battleship ship type definition."""

    name: str
    size: int


# Standard Battleship ship configuration
SHIPS = [
    ShipType("Carrier", 5),
    ShipType("Battleship", 4),
    ShipType("Cruiser", 3),
    ShipType("Submarine", 3),
    ShipType("Destroyer", 2),
]


class BattleshipEmulator:
    def __init__(self, seed=None):
        """Initialize Battleship emulator with standard 10x10 board and fixed ship configuration.

        Args:
            seed: Random seed for reproducible ship placement and game state.
        """
        # Board configuration - using standard Battleship constants
        self.cols = string.ascii_lowercase[:BOARD_SIZE]
        self.rows = [str(r) for r in range(1, BOARD_SIZE + 1)]

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

    def _generate_ship_coordinates(self, size, is_horizontal):
        """Generate coordinates for a ship of given size and orientation."""
        if is_horizontal:
            row = self.rng.choice(self.rows)
            start_idx = self.rng.randint(0, BOARD_SIZE - size)
            return [f"{self.cols[start_idx + i]}{row}" for i in range(size)]
        else:
            col = self.rng.choice(self.cols)
            start_idx = self.rng.randint(0, BOARD_SIZE - size)
            return [f"{col}{self.rows[start_idx + i]}" for i in range(size)]

    def _place_ships(self):
        """Place all ships on the board without overlapping."""
        occupied = set()

        for ship_type in SHIPS:
            placed = False
            while not placed:
                is_horizontal = self.rng.choice([True, False])
                coords = self._generate_ship_coordinates(ship_type.size, is_horizontal)

                # Check for overlap
                has_overlap = any(c in occupied for c in coords)
                if not has_overlap:
                    self.ships.append(
                        ShipStatus(
                            name=ship_type.name,
                            size=ship_type.size,
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

    def step(self, move: Coordinate) -> MoveOutcome:
        """
        Process a move and return the outcome.

        Args:
            move: Coordinate object representing the move

        Returns:
            MoveOutcome dataclass containing move result information
        """
        # Convert to string for internal board operations
        move_str = move.to_str()

        # Guard clause for invalid moves
        is_invalid_move = move_str not in self.board or self.board[move_str] != "?"
        if is_invalid_move:
            self.last_move_invalid = True
            self.history.append(move_str)
            self.turn_count += 1
            return MoveOutcome(move=move_str, hit=False, sunk=False, invalid=True, render=self.render())

        # Process valid move
        self.last_move_invalid = False

        # Check if move hits any ship
        hit_ship = None
        for ship in self.ships:
            if move_str in ship.coords:
                ship.hits.add(move_str)
                hit_ship = ship
                break

        # Determine move outcome
        hit = hit_ship is not None
        sunk = hit and hit_ship is not None and hit_ship.is_sunk

        # Update board based on outcome
        if hit and hit_ship:
            if sunk:
                # Mark all positions of the sunk ship as 's'
                for coord in hit_ship.coords:
                    self.board[coord] = "s"
            else:
                self.board[move_str] = "x"
        else:
            self.board[move_str] = "o"

        # Update game state
        self.history.append(move_str)
        self.turn_count += 1

        # Check for game over
        all_ships_sunk = all(ship.is_sunk for ship in self.ships)
        if all_ships_sunk:
            self.game_over = True

        return MoveOutcome(move=move_str, hit=hit, sunk=sunk, invalid=False, render=self.render())

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
        board_state = BoardState(board_size=BOARD_SIZE, cells=self.board.copy())

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

    def restore_state(self, game_state: GameState) -> None:
        """
        Restore emulator state from a GameState object.

        Args:
            game_state: GameState object to restore from
        """
        self.turn_count = game_state.turn_count
        self.game_over = game_state.game_over
        self.last_move_invalid = game_state.last_move_invalid
        self.history = game_state.history.copy()
        self.board = game_state.board.cells.copy()
        self.ships = [
            ShipStatus(name=ship.name, size=ship.size, coords=ship.coords.copy(), hits=ship.hits.copy())
            for ship in game_state.ships
        ]
