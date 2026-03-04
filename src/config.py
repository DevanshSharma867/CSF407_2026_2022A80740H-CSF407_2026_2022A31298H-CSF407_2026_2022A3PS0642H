"""
config.py
Centralized configuration for Oracle Agent — CS F407 AI Assignment
All parameters as separate variables (no hardcoding in other files).
"""

import random as _random

# ─────────────────────────────────────────────
#  Random Seed (set None for true randomness)
# ─────────────────────────────────────────────
RANDOM_SEED = None   # Change to int (e.g. 42) for reproducibility

if RANDOM_SEED is not None:
    _random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
#  Grid Dimensions
# ─────────────────────────────────────────────
GRID_ROWS = _random.randint(8, 10)
GRID_COLS = _random.randint(8, 10)

# ─────────────────────────────────────────────
#  Cell Type Constants
# ─────────────────────────────────────────────
CELL_LAND    = 'L'     # Safe land
CELL_VOLCANO = 'V'     # Hazard — costs 1 life
CELL_WATER   = 'W'     # Hazard — costs 1 life
CELL_WALL    = 'B'     # Brick wall — impassable
CELL_START   = 'S'     # Start (0,0)
CELL_GOAL    = 'G'     # Goal (M-1, N-1)

# ─────────────────────────────────────────────
#  Action Costs
# ─────────────────────────────────────────────
WALK_TIME_COST  = 2    # Time units consumed per walk (1 cell)
JUMP_TIME_COST  = 3    # Time units consumed per jump (2 cells)
TURN_COST       = 1    # Every action costs 1 Turn Unit
WALL_TURN_COST  = 1    # Bouncing off a wall costs 1 Turn Unit (no time cost)

# ─────────────────────────────────────────────
#  Agent Parameters
# ─────────────────────────────────────────────
AGENT_START_LIVES = 4  # Maximum lives
AGENT_MIN_LIVES   = 0  # Terminal failure threshold

# ─────────────────────────────────────────────
#  Grid Generation Parameters
# ─────────────────────────────────────────────
HAZARDS_ON_PATH    = 4          # Exactly 2V + 2W placed on ground-truth path
VOLCANOES_ON_PATH  = 2
WATERS_ON_PATH     = 2

# Distribution of non-path cells
NON_PATH_HAZARD_RATIO    = 0.50   # 50% hazardous (W or V)
NON_PATH_LAND_RATIO      = 0.40   # 40% safe land
NON_PATH_WALL_RATIO      = 0.10   # 10% brick wall

# ─────────────────────────────────────────────
#  Probabilistic Agent (Exercise 2)
# ─────────────────────────────────────────────
P_VOLCANO_BERNOULLI = 0.40    # Bernoulli param for volcano presence
P_WATER_BERNOULLI   = 0.40    # Bernoulli param for water presence

# Prior probabilities derived from Bernoulli params
PRIOR_VOLCANO = P_VOLCANO_BERNOULLI * (1 - P_WATER_BERNOULLI)   # 0.24
PRIOR_WATER   = (1 - P_VOLCANO_BERNOULLI) * P_WATER_BERNOULLI   # 0.24
PRIOR_LAND    = P_VOLCANO_BERNOULLI * P_WATER_BERNOULLI          # 0.16
PRIOR_WALL    = (1 - P_VOLCANO_BERNOULLI) * (1 - P_WATER_BERNOULLI)  # 0.36

# Conditional Probability Table: P(sensor=True | terrain)
# Format: {cell_type: (P_thermal_true, P_seismic_true)}
CPT = {
    CELL_VOLCANO: (0.85, 0.60),
    CELL_WATER:   (0.01, 0.85),
    CELL_LAND:    (0.10, 0.05),
    CELL_WALL:    (0.05, 0.01),
}

# Probabilistic agent scan settings
MAX_SCANS_PER_CELL  = 4      # After this, mark cell BLOCKED
RISK_SCAN_THRESHOLD = 0.35   # Scan if P(hazard) > this threshold
RISK_AVOID_THRESHOLD = 0.70  # Avoid cell entirely if P(hazard) > this

# ─────────────────────────────────────────────
#  Visualization Colors (matplotlib)
# ─────────────────────────────────────────────
COLOR_LAND    = '#90EE90'    # Light green
COLOR_VOLCANO = '#FF4500'    # Orange-red
COLOR_WATER   = '#4169E1'    # Royal blue
COLOR_WALL    = '#696969'    # Dim gray
COLOR_START   = '#FFD700'    # Gold
COLOR_GOAL    = '#32CD32'    # Lime green
COLOR_PATH    = '#FFA500'    # Orange highlight for path cells
COLOR_AGENT   = '#1E1E1E'    # Near black for agent marker

# Arrow colors
COLOR_WALK_ARROW   = '#000000'   # Black
COLOR_JUMP_ARROW   = '#800080'   # Purple
COLOR_DAMAGE_ARROW = '#FF0000'   # Red

# Belief map coloring thresholds (Ex2)
BELIEF_VOLCANO_THRESH = 0.40
BELIEF_WATER_THRESH   = 0.40
BELIEF_LAND_THRESH    = 0.50
COLOR_UNKNOWN         = '#AAAAAA'   # Gray for unknown cells
COLOR_SCAN_BORDER     = '#FFD700'   # Yellow highlight for scanned cells

# ─────────────────────────────────────────────
#  Output / File Settings
# ─────────────────────────────────────────────
OUTPUT_DIR              = 'outputs'
GROUND_TRUTH_PNG        = 'ground_truth_trajectory.png'
INITIAL_GRID_PNG        = 'initial_grid.png'
SIMULATION_GIF          = 'simulation.gif'
SIMULATION_MP4          = 'simulation.mp4'
FRAME_DIR               = 'frames'
GIF_FRAME_DURATION_MS   = 400   # milliseconds per frame in GIF
FIGURE_DPI              = 120
