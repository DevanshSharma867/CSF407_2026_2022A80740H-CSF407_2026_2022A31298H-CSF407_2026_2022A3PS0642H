"""
main.py
Oracle Agent — CS F407 AI Assignment
Entry point and orchestration layer.

Usage:
    python main.py [--exercise {1,2,both}] [--seed SEED]

Workflow:
  1. Load configuration from config.py
  2. Generate grid + ground-truth path  (grid_gen.py)
  3. Print cell IDs and grid to terminal
  4. Save ground-truth trajectory PNG and initial grid PNG
  5. Run agent search  (agent.py)
  6. Render and save animation  (visualizer.py)
"""

import os
import sys
import argparse
import config
import grid_gen
import visualizer
from agent import OracleAgent, ProbOracle


# ─────────────────────────────────────────────
#  Output directory setup
# ─────────────────────────────────────────────

def _setup_output_dirs(ex: int):
    base = os.path.join(config.OUTPUT_DIR, f'ex{ex}')
    os.makedirs(base, exist_ok=True)
    return base


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _print_banner(text: str):
    bar = '═' * 60
    print(f'\n{bar}')
    print(f'  {text}')
    print(f'{bar}')


def _print_step_log(steps, limit=20):
    print(f"\n  {'Step':<6} {'Action':<12} {'From':<10} {'To':<10}"
          f"{'Dmg':<6} {'Lives':<7} {'Turns':<7} {'Time':<7} {'Score':<8}")
    print(f"  {'─'*70}")
    for i, s in enumerate(steps[:limit]):
        dmg_str = '💀' if s.damage else '  '
        print(f"  {i+1:<6} {s.action:<12} {str(s.from_pos):<10} {str(s.to_pos):<10}"
              f"{dmg_str:<6} {s.lives_after:<7} {s.turns_after:<7} "
              f"{s.time_after:<7} {s.score:<8.2f}")
    if len(steps) > limit:
        print(f"  ... ({len(steps) - limit} more steps)")


def _print_final_report(steps):
    if not steps:
        print("\n  ❌  No path found — agent failed to reach the goal.")
        return
    last = steps[-1]
    print(f"\n  {'─'*50}")
    print(f"  FINAL REPORT")
    print(f"  {'─'*50}")
    print(f"  Total steps   : {len(steps)}")
    print(f"  Lives left    : {last.lives_after} / {config.AGENT_START_LIVES}")
    print(f"  Turns used    : {last.turns_after}")
    print(f"  Time units    : {last.time_after}")
    print(f"  Final score   : {last.score:.4f}  (lower = more optimal)")
    goal_reached = last.to_pos == (config.GRID_ROWS - 1, config.GRID_COLS - 1)
    status = '✅ GOAL REACHED' if goal_reached else '❌ FAILED'
    print(f"  Outcome       : {status}")
    print(f"  {'─'*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Exercise 1 — Deterministic
# ═══════════════════════════════════════════════════════════════════════════════

def run_exercise1(grid, path, anchors, hazard_map, out_dir):
    _print_banner('EXERCISE 1 — Deterministic Oracle Agent')

    # ── Agent search ────────────────────────────────────────────────────────
    print('\n  [*] Running A* search...')
    agent = OracleAgent(grid)
    steps = agent.search()

    if steps:
        print(f'  [✓] Path found in {len(steps)} steps.')
    else:
        print('  [✗] No valid path found.')

    _print_step_log(steps)
    _print_final_report(steps)

    # ── Visualise ─────────────────────────────────────────────────────────
    if steps:
        print('\n  [*] Rendering animation frames...')
        ex1_out = os.path.join(out_dir, 'ex1')
        os.makedirs(ex1_out, exist_ok=True)
        visualizer.animate_ex1(grid, steps, ex1_out)

    return steps


# ═══════════════════════════════════════════════════════════════════════════════
#  Exercise 2 — Probabilistic
# ═══════════════════════════════════════════════════════════════════════════════

def run_exercise2(grid, path, anchors, hazard_map, out_dir):
    _print_banner('EXERCISE 2 — Probabilistic Oracle Agent (Bayesian)')

    print(f'\n  Prior probabilities:')
    print(f'    P(Volcano) = {config.PRIOR_VOLCANO:.2f}')
    print(f'    P(Water)   = {config.PRIOR_WATER:.2f}')
    print(f'    P(Land)    = {config.PRIOR_LAND:.2f}')
    print(f'    P(Wall)    = {config.PRIOR_WALL:.2f}')

    # ── Agent search ────────────────────────────────────────────────────────
    print('\n  [*] Running Bayesian A* search...')
    prob_agent = ProbOracle(grid)
    steps = prob_agent.search()

    if steps:
        print(f'  [✓] Path found in {len(steps)} steps '
              f'(including {sum(1 for s in steps if s.action == "scan")} scan steps).')
    else:
        print('  [✗] No valid path found.')

    _print_step_log(steps)
    _print_final_report(steps)

    # ── Visualise ─────────────────────────────────────────────────────────
    if steps:
        print('\n  [*] Rendering dual-panel animation frames...')
        ex2_out = os.path.join(out_dir, 'ex2')
        os.makedirs(ex2_out, exist_ok=True)
        visualizer.animate_ex2(grid, steps, ex2_out)

    return steps


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Oracle Agent — CS F407 AI Assignment')
    parser.add_argument('--exercise', choices=['1', '2', 'both'],
                        default='both', help='Which exercise to run')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Override seed from CLI if provided
    if args.seed is not None:
        import random as _r
        config.RANDOM_SEED = args.seed
        _r.seed(args.seed)
        # Re-draw grid dimensions now that seed is set
        config.GRID_ROWS = _r.randint(8, 10)
        config.GRID_COLS = _r.randint(8, 10)

    out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Configuration summary ─────────────────────────────────────
    _print_banner('ORACLE AGENT — CS F407 AI Assignment')
    print(f'\n  Configuration:')
    print(f'    Grid Size         : {config.GRID_ROWS} × {config.GRID_COLS}')
    print(f'    Agent Start Lives : {config.AGENT_START_LIVES}')
    print(f'    Walk cost         : {config.WALK_TIME_COST} time + {config.TURN_COST} turn')
    print(f'    Jump cost         : {config.JUMP_TIME_COST} time + {config.TURN_COST} turn')
    print(f'    Random seed       : {config.RANDOM_SEED}')

    # ── Step 2: Generate grid ──────────────────────────────────────────────
    _print_banner('GRID GENERATION')
    print('\n  [*] Generating grid...')
    grid = path = anchors = hazard_map = None
    for attempt in range(10):
        try:
            grid, path, anchors, hazard_map = grid_gen.generate_grid()
            break
        except RuntimeError as e:
            print(f"  [!] Grid gen attempt {attempt+1} failed: {e}, retrying...")
    else:
        print("  [✗] Grid generation failed after 10 attempts. Try a different seed.")
        sys.exit(1)
    a1, a2 = anchors
    print(f'  [✓] Grid generated  ({config.GRID_ROWS}×{config.GRID_COLS})')
    print(f'  [✓] Ground-truth path length : {len(path)} cells')
    print(f'  [✓] Anchor cells : A1={a1}, A2={a2}')
    print(f'  [✓] Path hazards : {hazard_map}')

    # ── Step 3: Print cell IDs ────────────────────────────────────────────
    grid_gen.print_cell_ids(grid)

    # ── Step 4: Print initial grid ────────────────────────────────────────
    grid_gen.print_grid(grid, path=path, label='Initial Grid  ([ ] = path cell)')

    # ── Step 5: Save static PNGs ──────────────────────────────────────────
    _print_banner('SAVING STATIC PLOTS')
    gt_png   = os.path.join(out_dir, config.GROUND_TRUTH_PNG)
    init_png = os.path.join(out_dir, config.INITIAL_GRID_PNG)
    visualizer.plot_ground_truth(grid, path, anchors, gt_png)
    visualizer.plot_initial_grid(grid, init_png)

    # ── Step 6: Run exercises ─────────────────────────────────────────────
    run_ex1 = args.exercise in ('1', 'both')
    run_ex2 = args.exercise in ('2', 'both')

    if run_ex1:
        run_exercise1(grid, path, anchors, hazard_map, out_dir)

    if run_ex2:
        run_exercise2(grid, path, anchors, hazard_map, out_dir)

    _print_banner('ALL DONE')
    print(f'  Output files are in: {os.path.abspath(out_dir)}\n')


if __name__ == '__main__':
    main()
