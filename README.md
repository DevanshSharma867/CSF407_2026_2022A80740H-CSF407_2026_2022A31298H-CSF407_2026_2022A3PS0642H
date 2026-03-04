# Oracle Agent — CS F407 Artificial Intelligence
## Project Assignment I — Mid-Semester Evaluative

An intelligent agentic system that navigates a hazardous grid environment in two modes:
- **Exercise 1:** Deterministic agent with full grid visibility and A\* search
- **Exercise 2:** Probabilistic agent using Bayesian inference and noisy sensors

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Environment Setup](#environment-setup)
3. [Running the Project](#running-the-project)
4. [Command-Line Options](#command-line-options)
5. [Grid Environment](#grid-environment)
6. [Exercise 1 — Deterministic Oracle Agent](#exercise-1--deterministic-oracle-agent)
7. [Exercise 2 — Probabilistic Oracle Agent](#exercise-2--probabilistic-oracle-agent)
8. [Grid Generation Algorithm](#grid-generation-algorithm)
9. [Visualisation Guide](#visualisation-guide)
10. [Output Files](#output-files)
11. [Configuration Reference](#configuration-reference)
12. [Sample Results](#sample-results)
13. [Design Decisions & Cost Model](#design-decisions--cost-model)

---

## Repository Structure

```
CSF407_2026_ID_Assignment-I/
├── README.md                        ← This file
├── config.yml                       ← Conda environment definition
└── src/
    ├── config.py                    ← All parameters as separate variables
    ├── grid_gen.py                  ← Unique path + grid generation
    ├── agent.py                     ← OracleAgent (Ex1) + ProbOracle (Ex2)
    ├── visualizer.py                ← Static plots + animated simulation
    ├── main.py                      ← Entry point and orchestration
    └── outputs/                     ← Generated automatically on first run
        ├── ground_truth_trajectory.png
        ├── initial_grid.png
        ├── ex1/
        │   ├── frames/              ← Per-step PNG frames
        │   ├── simulation.gif
        │   └── simulation.mp4
        └── ex2/
            ├── frames/
            ├── simulation.gif
            └── simulation.mp4
```

---

## Environment Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- `ffmpeg` (required for MP4 output — GIF works without it)

### Install ffmpeg (optional but recommended)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows (with conda)
conda install -c conda-forge ffmpeg
```

### Create the Conda Environment

```bash
# From the project root directory (where config.yml lives)
conda env create -f config.yml

# Activate the environment
conda activate CSF407_2025_Oracle
```

The `config.yml` installs:
- Python 3.10
- numpy, matplotlib, scipy, imageio
- All dependencies needed to run both exercises

### Verify Installation

```bash
conda activate CSF407_2025_Oracle
cd src
python -c "import numpy, matplotlib, imageio; print('All dependencies OK')"
```

---

## Running the Project

All commands must be run from inside the `src/` directory with the conda environment active.

```bash
conda activate CSF407_2025_Oracle
cd src
```

### Run both exercises (default)

```bash
python main.py
```

### Run with a fixed random seed (for reproducibility)

```bash
python main.py --seed 42
```

> **Note:** Without `--seed`, a new unique grid is generated every run. Use `--seed` to reproduce a specific result for demonstration or debugging.

### Run only Exercise 1

```bash
python main.py --exercise 1
```

### Run only Exercise 2

```bash
python main.py --exercise 2
```

### Combine options

```bash
python main.py --exercise 1 --seed 99
python main.py --exercise both --seed 7
```

---

## Command-Line Options

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--exercise` | `1`, `2`, `both` | `both` | Which exercise(s) to run |
| `--seed` | any integer | `None` (random) | Random seed for grid generation |

---

## Grid Environment

The grid is **M × N** with 8 ≤ M, N ≤ 10 (default **9 × 9**, set in `config.py`).

### Cell Types

| Symbol | Type | Description |
|--------|------|-------------|
| `S` | Start | Agent spawn point — always at (0, 0) |
| `G` | Goal | Target cell — always at (M-1, N-1) |
| `L` | Land | Safe to traverse, no penalty |
| `V` | Volcano | Hazardous — agent loses 1 life on entry |
| `W` | Water | Hazardous — agent loses 1 life on entry |
| `B` | Brick Wall | Impassable obstacle |

### Movement Rules

| Action | Distance | Time Cost | Turn Cost | Notes |
|--------|----------|-----------|-----------|-------|
| Walk | 1 cell (orthogonal) | +2 | +1 | Can move into any non-wall cell |
| Jump | 2 cells (orthogonal) | +3 | +1 | Skips middle cell; cannot jump *over* a Brick Wall |
| Wall Bounce | 0 cells | 0 | +1 | Agent tried to enter/jump over a wall; stays in place |

**Lives:** Agent starts with 4 lives. Stepping on Volcano or Water costs 1 life. Reaching 0 lives is terminal failure. After taking damage the agent resumes from the same cell.

---

## Exercise 1 — Deterministic Oracle Agent

### Objective

Navigate from Start `(0,0)` to Goal `(M-1, N-1)` while **optimising the score**:

```
Score = (Turn Units + Time Units) / Lives Remaining
```

Lower score = better (faster path with more lives preserved). The agent aims to reach the goal at the fastest speed possible while minimising score.

### Sensors (Deterministic)

At every cell the agent reads two deterministic sensors based on **orthogonal neighbours**:
- **Thermal sensor:** `True` if any adjacent cell is a Volcano
- **Seismic sensor:** `True` if any adjacent cell is Water

These are perfect (noise-free) in Exercise 1.

### Search Algorithm — A\*

**State:** `(row, col, lives)`

**Cost function:**
```
g(n) = turns_so_far + time_so_far
h(n) = Manhattan_distance × WALK_TIME_COST    ← admissible lower bound
f(n) = (g(n) + h(n)) / lives_remaining        ← unified scaling, admissible
```

Dividing the entire `(g + h)` estimate by lives ensures the heuristic remains admissible and consistent, and directly reflects the assignment's score objective.

**Wall bounce:** When the agent attempts to walk into or jump over a Brick Wall, it pays 1 Turn Unit and stays in place. This state is pushed back onto the A\* heap so the search tree remains complete.

**Tie-breaking:** States with equal `f` values are resolved by turns, then time, then lives (descending).

---

## Exercise 2 — Probabilistic Oracle Agent

### Objective

Same score metric as Exercise 1. The agent is **blind** — it cannot directly observe cell types and must reason using noisy sensor readings and Bayesian inference.

### Prior Beliefs

Derived from independent Bernoulli distributions with `p_volcano = 0.4`, `p_water = 0.4`:

| Cell Type | Prior Probability | Derivation |
|-----------|-------------------|------------|
| Volcano   | **0.24** | p_v × (1 − p_w) = 0.4 × 0.6 |
| Water     | **0.24** | (1 − p_v) × p_w = 0.6 × 0.4 |
| Land      | **0.16** | p_v × p_w = 0.4 × 0.4 |
| Wall      | **0.36** | (1 − p_v) × (1 − p_w) = 0.6 × 0.6 |

### Conditional Probability Table (CPT)

| Cell Type | P(Thermal = True \| Terrain) | P(Seismic = True \| Terrain) |
|-----------|------------------------------|------------------------------|
| Volcano   | 0.85 | 0.60 |
| Water     | 0.01 | 0.85 |
| Land      | 0.10 | 0.05 |
| Wall      | 0.05 | 0.01 |

### Bayesian Update Rule

Each scan of cell `(r, c)` fires both sensors and updates the belief via Bayes' theorem (sensors assumed conditionally independent given terrain):

```
P(type | obs) ∝ P(thermal_obs | type) × P(seismic_obs | type) × P(type)
```

Posteriors are normalised to sum to 1.

### Scan Rules

- Each scan advances the **turn counter by 1** (no movement)
- Maximum **4 scans per cell** — after that the cell is marked `BLOCKED` for scanning (but remains physically walkable if belief permits)
- The agent scans the next planned cell only if `P(hazard) > RISK_SCAN_THRESHOLD` (default 0.35) **and** the cell has not already been scanned in the current planning cycle (prevents spin-loop)

### Dynamic Replanning (Sense–Plan–Act Loop)

The agent does **not** pre-plan a full path and execute it blindly. Instead, at every step:

1. **Replan:** Run A\* from current position using current belief grid (no ground-truth access)
2. **Scan:** If the next planned cell has `P(hazard) > 0.35`, scan it once and replan
3. **Execute:** Take one step toward the next cell
4. **Observe:** Receive true outcome (damage or safe arrival); collapse cell belief to certainty
5. **Repeat** until goal reached or lives exhausted

This ensures the agent always acts on the most up-to-date information.

### Ground-Truth Isolation

The planning module (`_plan`) accesses **only** `self.belief` — never `self.grid`. The only legal uses of ground truth are:
- `_noisy_read()` — physically simulates sensor firing based on true terrain
- `_execute_step()` — resolves whether the agent actually hits a wall or takes damage

### Goal and Start Cell Initialisation

Both the start cell `(0,0)` and goal cell `(M-1, N-1)` are initialised as **known Land** at agent creation. This is semantically justified: the agent is handed these coordinates as part of the problem definition, so it knows they are reachable safe cells even without prior exploration.

---

## Grid Generation Algorithm

Every run produces a **unique, reproducible path** through a fresh grid.

### Step 1 — Anchor Selection

Two intermediate anchor cells are chosen using a **diagonal-zone strategy**:
- **A1:** Random cell in the upper-left zone (rows 1 to M//2−1, cols 1 to N//2−1)
- **A2:** Random cell in the lower-right zone (rows M//2+1 to M−2, cols N//2+1 to N−2)

This geometry **guarantees** A1 and A2 are non-adjacent and share neither row nor column, satisfying the spec constraint without any retry logic.

### Step 2 — Randomised Manhattan Walk

Three path segments are stitched together: Start → A1 → A2 → Goal.

Each segment uses a **randomised Manhattan walk**: at every step, the axis to move along (row or column) is chosen randomly among directions that reduce Manhattan distance. This produces a different corridor through the grid on every run while always making forward progress.

### Step 3 — Hazard Placement on Path

Exactly **2 Volcanoes** and **2 Waters** are placed on interior path cells (never on Start or Goal). Placement enforces a minimum gap of 2 path positions between any two hazards, ensuring no two hazards are adjacent along the path.

### Step 4 — Non-Path Cell Distribution

All cells not on the ground-truth path are filled randomly:
- **50%** hazardous (Volcano or Water, split equally)
- **40%** Land
- **10%** Brick Wall

The resulting grid satisfies the spec: if the path has K cells, then K−4 interior path cells are Land, and the non-path distribution matches the stated ratios.

---

## Visualisation Guide

### Exercise 1 — Single-Panel Animation

Each frame shows the **ground-truth grid** with all arrows drawn up to that step.

| Element | Appearance |
|---------|------------|
| Walk (no damage) | Black solid arrow |
| Jump (no damage) | Purple curved arrow |
| Walk or Jump with damage | **Red dashed arrow** |
| Wall bounce attempt | Orange dotted stub arrow with "B" label |
| Agent position | Black filled circle |
| Damage marker | Red ✗ at the cell where life was lost (persists all frames) |
| HUD | Step / Lives / Turns / Time / Score at top of frame |

### Exercise 2 — Dual-Panel Animation

**Left panel — Belief Map (Agent's Perspective):**

| Colour | Condition |
|--------|-----------|
| Red | `P(Volcano) > 0.40` |
| Blue | `P(Water) > 0.40` |
| Green | `P(Land) > 0.50` |
| Gray | Unknown (none of the above) |

Additional features:
- Risk percentage `P(hazard)` displayed in each cell
- Yellow border on all cells that have been scanned (persists)
- `T:T/F  S:T/F` sensor reading labels on all scanned cells (persists across frames)
- Red ✗ at cells where agent took damage
- Agent shown as black circle on **every frame**, including scan steps

**Right panel — Ground Truth:**
- True cell colours (Red=Volcano, Blue=Water, Green=Land, Gray=Wall)
- Cell labels (V / W / L / B)
- Same path arrows as left panel for comparison
- Red ✗ damage markers

---

## Output Files

All outputs are written to `src/outputs/` automatically.

| File | Description |
|------|-------------|
| `outputs/ground_truth_trajectory.png` | Static plot of the A1→A2 path with arrow overlays and legend |
| `outputs/initial_grid.png` | Static full-grid view with cell type colours |
| `outputs/ex1/frames/frame_XXXX.png` | Per-step animation frames (Exercise 1) |
| `outputs/ex1/simulation.gif` | Compiled GIF animation (Exercise 1) |
| `outputs/ex1/simulation.mp4` | Compiled MP4 video (Exercise 1, requires ffmpeg) |
| `outputs/ex2/frames/frame_XXXX.png` | Per-step dual-panel frames (Exercise 2) |
| `outputs/ex2/simulation.gif` | Compiled GIF animation (Exercise 2) |
| `outputs/ex2/simulation.mp4` | Compiled MP4 video (Exercise 2, requires ffmpeg) |

Frame rate: configurable via `GIF_FRAME_DURATION_MS` in `config.py` (default 400 ms/frame = 2.5 fps).

---

## Configuration Reference

All parameters live in `src/config.py` as separate named variables. Key settings:

### Grid

```python
GRID_ROWS = 9          # M — must be 8 ≤ M ≤ 10
GRID_COLS = 9          # N — must be 8 ≤ N ≤ 10
RANDOM_SEED = None     # None = random each run; set int for reproducibility
```

### Action Costs

```python
WALK_TIME_COST = 2     # Time units per walk
JUMP_TIME_COST = 3     # Time units per jump
TURN_COST      = 1     # Every action costs 1 turn
WALL_TURN_COST = 1     # Wall bounce: 1 turn, no time cost
```

### Agent

```python
AGENT_START_LIVES = 4  # Starting lives
```

### Probabilistic Agent (Exercise 2)

```python
P_VOLCANO_BERNOULLI  = 0.40   # Bernoulli parameter for volcano
P_WATER_BERNOULLI    = 0.40   # Bernoulli parameter for water
MAX_SCANS_PER_CELL   = 4      # Scan limit per cell
RISK_SCAN_THRESHOLD  = 0.35   # Scan next cell if P(hazard) exceeds this
```

### Visualisation

```python
GIF_FRAME_DURATION_MS = 400   # Milliseconds per frame
FIGURE_DPI            = 120   # Output image resolution
```

---

## Sample Results

Running with `--seed 7` on a 9×9 grid:

```
Grid generated (9×9)
Ground-truth path length : 17 cells
Anchor cells : A1=(2, 1),  A2=(6, 7)
Path hazards : {(0,1):'V', (2,1):'W', (2,3):'W', (3,4):'V'}
```

**Exercise 1 — Deterministic Agent**

```
Step  Action  From    To      Dmg  Lives  Turns  Time  Score
1     jump    (0,0)   (2,0)   ✗    3      1      3     1.33
2     jump    (2,0)   (2,2)        3      2      6     2.67
3     jump    (2,2)   (2,4)        3      3      9     4.00
4     walk    (2,4)   (2,5)        3      4      11    5.00
5     jump    (2,5)   (2,7)        3      5      14    6.33
6     jump    (2,7)   (4,7)        3      6      17    7.67
7     jump    (4,7)   (6,7)        3      7      20    9.00
8     jump    (6,7)   (8,7)        3      8      23    10.33
9     walk    (8,7)   (8,8)        3      9      25    11.33

Lives: 3/4 | Turns: 9 | Time: 25 | Final Score: 11.33 | ✅ GOAL REACHED
```

**Exercise 2 — Probabilistic Agent**

```
19 steps (9 movement + 9 scans + 1 wall bounce)
Lives: 1/4 | Turns: 19 | Time: 25 | Final Score: 44.00 | ✅ GOAL REACHED
```

---

## Design Decisions & Cost Model

### Score Objective

The assignment specifies:

> Score = (Turn Units + Time Units) / Lives Remaining

This is **minimised** (not maximised) — a lower score means the agent was faster *and* preserved more lives. The professor confirmed this: *"The wording should be 'Optimizes the score at fastest speed possible to reach goal state' instead of 'Maximizes'."*

### A\* Heuristic Admissibility

```
g(n)  = turns_so_far + time_so_far        ← exact assignment objective numerator
h(n)  = Manhattan_distance × WALK_TIME_COST  ← conservative lower bound (same units as g)
f(n)  = (g(n) + h(n)) / lives            ← full estimate divided by lives (not just g)
```

Dividing the **entire** `(g + h)` by lives is critical. If only `g` were divided, the heuristic term `h` would be in a different scale than `g/lives`, breaking admissibility. With unified scaling, `h` remains a valid lower bound on the remaining score contribution.

### Probabilistic Edge Cost

```
edge_cost(cell, action) = base_cost + P(hazard | belief) × LIFE_HAZARD_WEIGHT
```

`LIFE_HAZARD_WEIGHT = 6` approximates the score penalty of losing one life. Derivation:

> Losing one life (L → L−1) increases score by:
> ΔS = (turns + time) / (L × (L−1))
>
> For mid-game values (turns+time ≈ 12, L ≈ 3):
> ΔS ≈ 12 / (3×2) = 2 turns (minimum)
>
> The remaining journey also costs more at L−1 lives, so a conservative
> multiplier of 6 captures the expected full lifetime impact.

This is an **expected-value approximation** that avoids building a full POMDP tree while producing rational, risk-aware behaviour.

### Why Not a Full POMDP?

A proper POMDP would require expanding the belief state space at every node, making planning exponentially expensive for a 9×9 grid with 4 cell types. The chosen approach — A\* over the mean belief with a risk penalty — runs in milliseconds and produces behaviour that is demonstrably sensible: it scans uncertain cells before entering, avoids high-probability hazards, and replans dynamically after every observation.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'imageio'`**
→ Make sure the conda environment is active: `conda activate CSF407_2025_Oracle`

**GIF created but no MP4**
→ `ffmpeg` is not installed or not on PATH. Install it (see [Environment Setup](#environment-setup)). The GIF is always created regardless.

**Agent stops short of the goal**
→ Should not happen in the current implementation. If encountered, run with `--seed 42` to confirm. The goal cell is pre-initialised as known-safe so it is never misidentified as a wall.

**Different results each run**
→ By design — each run generates a unique grid. Use `--seed N` for a fixed grid.

**Very high Exercise 2 score**
→ Expected on difficult grids where the agent takes several hazard hits. The probabilistic agent relies on noisy sensors and cannot perfectly avoid every hazard. Score depends on the random grid and sensor readings for a given seed.
