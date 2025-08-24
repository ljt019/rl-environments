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
