"""
grid_gen.py
Grid generation for Oracle Agent — CS F407 AI Assignment

Algorithm:
  1. Pick two anchor cells using full-grid rejection sampling (Bug 9).
  2. Connect Start→A1→A2→Goal using a randomised Manhattan walk:
       At each step, randomly choose horizontal or vertical movement (with shuffled direction
       preference), giving a different corridor every run.
  3. Place exactly 2 Volcanoes and 2 Waters on the path (non-adjacent to each other).
     Anchors are protected (Bug 2).
  4. Fill remaining non-path cells: 50% hazardous, 40% land, 10% wall.
"""

import random
import config


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _random_walk_path(start: tuple, end: tuple, forbidden: set) -> list:
    """
    Build a path from `start` to `end` using a randomised Manhattan walk.
    At each step, prefer a random axis (row or col); if blocked, use the other.
    Returns list of (row, col) tuples INCLUDING start but EXCLUDING end
    (so caller can chain segments without duplicating waypoints).
    """
    path = [start]
    r, c = start
    er, ec = end

    max_steps = (config.GRID_ROWS + config.GRID_COLS) * 4
    steps = 0

    while (r, c) != (er, ec) and steps < max_steps:
        steps += 1
        dr = er - r
        dc = ec - c

        # Randomly decide which axis to move on; bias toward remaining distance
        choices = []
        if dr != 0:
            choices.append(('r', 1 if dr > 0 else -1))
        if dc != 0:
            choices.append(('c', 1 if dc > 0 else -1))

        # Shuffle to introduce randomness
        random.shuffle(choices)

        moved = False
        for axis, delta in choices:
            if axis == 'r':
                nr, nc = r + delta, c
            else:
                nr, nc = r, c + delta

            if 0 <= nr < config.GRID_ROWS and 0 <= nc < config.GRID_COLS:
                if (nr, nc) not in forbidden:  # Bug 9: respect forbidden cells
                    r, c = nr, nc
                    if (r, c) not in path:
                        path.append((r, c))
                    moved = True
                    break

        if not moved:
            break

    return path  # does NOT include `end`; caller appends next segment


def _build_full_path(start, a1, a2, goal, max_attempts=500):
    """
    Concatenate three segments: start→a1, a1→a2, a2→goal.
    Bug 10: Added retry logic and RuntimeError.
    Bug 3: Use empty forbidden sets to increase success rate.
    """
    for attempt in range(max_attempts):
        # Bug 3: Use empty forbidden sets for segments
        seg1 = _random_walk_path(start, a1, set())
        # The function returns path excluding end. Let's check if we can reach end from last path element.
        lr, lc = seg1[-1]
        if abs(lr - a1[0]) + abs(lc - a1[1]) != 1 and (lr, lc) != a1:
            continue
        
        seg2 = _random_walk_path(a1, a2, set())
        lr, lc = seg2[-1]
        if abs(lr - a2[0]) + abs(lc - a2[1]) != 1 and (lr, lc) != a2:
            continue
            
        seg3 = _random_walk_path(a2, goal, set())
        lr, lc = seg3[-1]
        if abs(lr - goal[0]) + abs(lc - goal[1]) != 1 and (lr, lc) != goal:
            continue

        # Combine, appending junction points correctly
        # Since _random_walk_path excludes end, we append the junctions manually
        full = seg1 + [a1] + seg2[1:] + [a2] + seg3[1:] + [goal]
        # Remove any potential adjacent duplicates if any segment was empty or start==end
        unique_full = []
        for p in full:
            if not unique_full or p != unique_full[-1]:
                unique_full.append(p)
        
        if unique_full[-1] == goal:
            return unique_full

    raise RuntimeError(f"Failed to build a connected path after {max_attempts} attempts.")


def _place_hazards_on_path(path: list, a1: tuple, a2: tuple) -> dict:
    """
    Place exactly 2 Volcanoes and 2 Waters on interior path cells
    (never on start or goal or anchors), ensuring no two hazards are adjacent in the path list.

    Returns dict: {(r,c): cell_type}
    """
    protected = {path[0], path[-1], a1, a2} # Bug 2: protect anchors
    interior = [cell for cell in path if cell not in protected]
    
    if len(interior) < 4:
        raise ValueError("Path too short to place 4 hazards after protecting anchor cells.")

    hazard_types = [config.CELL_VOLCANO] * config.VOLCANOES_ON_PATH + \
                   [config.CELL_WATER] * config.WATERS_ON_PATH
    random.shuffle(hazard_types)

    # Re-using the logic from interior but we need the indices in the full path for adjacency check
    path_indices = [i for i, cell in enumerate(path) if cell in interior]
    
    assigned = {}
    last_hazard_idx = -2

    for idx in path_indices:
        if len(assigned) == len(hazard_types):
            break
        if idx - last_hazard_idx >= 2:
            assigned[path[idx]] = hazard_types[len(assigned)]
            last_hazard_idx = idx

    if len(assigned) < 4:
        # Relaxed retry
        assigned = {}
        candidates = path_indices[:]
        random.shuffle(candidates)
        placed_indices = []
        for idx in candidates:
            too_close = any(abs(idx - p) < 2 for p in placed_indices)
            if not too_close:
                assigned[path[idx]] = hazard_types[len(assigned)]
                placed_indices.append(idx)
            if len(assigned) == 4:
                break

    return assigned


def _fill_non_path_cells(grid: list, path_set: set):
    """
    Fill all non-path cells according to distribution:
      50% hazardous (V or W equally split), 40% Land, 10% Wall
    """
    non_path = [
        (r, c)
        for r in range(config.GRID_ROWS)
        for c in range(config.GRID_COLS)
        if (r, c) not in path_set
    ]
    random.shuffle(non_path)

    n = len(non_path)
    n_hazard = round(n * config.NON_PATH_HAZARD_RATIO)
    n_land   = round(n * config.NON_PATH_LAND_RATIO)
    # wall gets the rest
    n_wall   = n - n_hazard - n_land

    # Split hazard equally between V and W
    half_h = n_hazard // 2
    cell_types = (
        [config.CELL_VOLCANO] * half_h +
        [config.CELL_WATER]   * (n_hazard - half_h) +
        [config.CELL_LAND]    * n_land +
        [config.CELL_WALL]    * n_wall
    )
    random.shuffle(cell_types)

    for (r, c), ct in zip(non_path, cell_types):
        grid[r][c] = ct


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def generate_grid():
    """
    Generate a unique MxN grid with a guaranteed safe path.

    Returns
    -------
    grid        : list[list[str]]  — 2-D grid of cell types
    path        : list[tuple]      — ground-truth trajectory [(r,c), ...]
    anchors     : (a1, a2)        — the two intermediate anchor cells
    hazard_map  : dict             — {(r,c): 'V'|'W'} hazards placed on path
    """
    M, N = config.GRID_ROWS, config.GRID_COLS
    start = (0, 0)
    goal  = (M - 1, N - 1)

    # ── 1. Pick anchor cells (Bug 9: Rejection Sampling) ──────────────────
    candidates = [
        (r, c)
        for r in range(M)
        for c in range(N)
        if (r, c) not in {start, goal}
    ]

    a1, a2 = (0, 0), (0, 0)
    found_anchors = False
    for _ in range(50_000):
        a1, a2 = random.sample(candidates, 2)
        if a1[0] == a2[0]: continue          # must not share row
        if a1[1] == a2[1]: continue          # must not share column
        if abs(a1[0]-a2[0]) + abs(a1[1]-a2[1]) <= 1: continue  # must not be adjacent
        found_anchors = True
        break
    
    if not found_anchors:
        raise RuntimeError("Failed to find suitable anchor cells.")

    # ── 2. Build randomised path ──────────────────────────────────────────
    path = _build_full_path(start, a1, a2, goal)

    path_set = set(path)

    # ── 3. Place hazards on path ──────────────────────────────────────────
    hazard_map = _place_hazards_on_path(path, a1, a2)

    # ── 4. Build grid ─────────────────────────────────────────────────────
    grid = [[config.CELL_LAND] * N for _ in range(M)]

    # Mark start and goal
    grid[start[0]][start[1]] = config.CELL_START
    grid[goal[0]][goal[1]]   = config.CELL_GOAL

    # Mark path cells (Land unless hazard)
    for (r, c) in path:
        if (r, c) == start or (r, c) == goal:
            continue
        if (r, c) in hazard_map:
            grid[r][c] = hazard_map[(r, c)]
        else:
            grid[r][c] = config.CELL_LAND

    # ── 5. Fill non-path cells ─────────────────────────────────────────────
    _fill_non_path_cells(grid, path_set)

    # Re-stamp start/goal (fill may have overwritten them)
    grid[start[0]][start[1]] = config.CELL_START
    grid[goal[0]][goal[1]]   = config.CELL_GOAL

    return grid, path, (a1, a2), hazard_map


def print_grid(grid: list, path: list = None, label: str = "Grid"):
    """Pretty-print the grid to the terminal with optional path highlight."""
    M = len(grid)
    N = len(grid[0]) if M > 0 else 0
    path_set = set(path) if path else set()

    print(f"\n{'═'*50}")
    print(f"  {label}  ({M}×{N})")
    print(f"{'═'*50}")

    # Column header
    print("     " + "  ".join(f"{c:2d}" for c in range(N)))
    print("    " + "─" * (N * 4))

    for r in range(M):
        row_str = f"{r:3d} │ "
        for c in range(N):
            cell = grid[r][c]
            marker = f"[{cell}]" if (r, c) in path_set else f" {cell} "
            row_str += marker
        print(row_str)

    print(f"{'═'*50}\n")


def print_cell_ids(grid: list):
    """Print cell IDs (row,col) for every cell."""
    M = len(grid)
    N = len(grid[0])
    print(f"\n{'─'*60}")
    print("  CELL IDs  (row, col) → type")
    print(f"{'─'*60}")
    for r in range(M):
        for c in range(N):
            print(f"  ({r},{c}) → {grid[r][c]}", end="   ")
        print()
    print(f"{'─'*60}\n")
