"""
visualizer.py
Visualizer for Oracle Agent — CS F407 AI Assignment

Functions
---------
plot_ground_truth(grid, path, anchors, out_path)
    Static plot of ground-truth trajectory.

plot_initial_grid(grid, out_path)
    Static plot of initial grid.

animate_ex1(grid, steps, out_dir)
    Animated visualisation for Exercise 1 (single panel).

animate_ex2(grid, steps, out_dir)
    Animated dual-panel visualisation for Exercise 2.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

import config
from agent import Step

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


_CELL_COLORS = {
    config.CELL_LAND: config.COLOR_LAND,
    config.CELL_VOLCANO: config.COLOR_VOLCANO,
    config.CELL_WATER: config.COLOR_WATER,
    config.CELL_WALL: config.COLOR_WALL,
    config.CELL_START: config.COLOR_START,
    config.CELL_GOAL: config.COLOR_GOAL,
}


def _cell_color(cell_type):
    return _CELL_COLORS.get(cell_type, '#FFFFFF')


def _belief_color(probs):
    """Color a belief-map cell based on posterior probabilities."""
    pv = probs.get(config.CELL_VOLCANO, 0)
    pw = probs.get(config.CELL_WATER, 0)
    pl = probs.get(config.CELL_LAND, 0)
    if pv > config.BELIEF_VOLCANO_THRESH:
        return config.COLOR_VOLCANO
    if pw > config.BELIEF_WATER_THRESH:
        return config.COLOR_WATER
    if pl > config.BELIEF_LAND_THRESH:
        return config.COLOR_LAND
    return config.COLOR_UNKNOWN


def _draw_grid_cells(ax, grid, path_set=None, highlight_cells=None):
    """Draw all cells of the grid on ax."""
    M = len(grid)
    N = len(grid[0])
    path_set = path_set or set()
    highlight_cells = highlight_cells or set()

    for r in range(M):
        for c in range(N):
            ct = grid[r][c]
            color = _cell_color(ct)
            if (r, c) in path_set:
                color = config.COLOR_PATH

            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                 color=color, ec='#333333', lw=0.8)
            ax.add_patch(rect)

            if (r, c) in highlight_cells:
                rect2 = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                      fill=False, ec='yellow', lw=3)
                ax.add_patch(rect2)

            ax.text(c, r, ct, ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])


def _draw_belief_cells(ax, belief_snap, scan_cells=None, damage_cells=None):
    """Draw belief-map cells (left panel of Ex2)."""
    M = config.GRID_ROWS
    N = config.GRID_COLS
    scan_cells = scan_cells or set()
    damage_cells = damage_cells or set()

    for r in range(M):
        for c in range(N):
            probs = belief_snap.get((r, c), {})
            color = _belief_color(probs)
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                 color=color, ec='#333333', lw=0.8)
            ax.add_patch(rect)

            if (r, c) in scan_cells:
                rect2 = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                      fill=False, ec=config.COLOR_SCAN_BORDER, lw=3)
                ax.add_patch(rect2)

            if (r, c) in damage_cells:
                ax.text(c, r, 'X', ha='center', va='center',
                        fontsize=14, color='red', fontweight='bold')

            p_haz = probs.get(config.CELL_VOLCANO, 0) + probs.get(config.CELL_WATER, 0)
            ax.text(c, r - 0.28, f'{p_haz*100:.0f}%', ha='center', va='center',
                    fontsize=6, color='white',
                    path_effects=[pe.withStroke(linewidth=1, foreground='black')])


def _setup_ax(ax, M, N, title):
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(M - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(N))
    ax.set_yticks(range(M))
    ax.set_xticklabels([str(i) for i in range(N)], fontsize=7)
    ax.set_yticklabels([str(i) for i in range(M)], fontsize=7)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.grid(False)


def _draw_arrow(ax, from_pos, to_pos, action, damage, alpha=1.0):
    """Draw a movement arrow on ax representing the agent's action."""
    r1, c1 = from_pos
    r2, c2 = to_pos

    if damage:
        arrow_color = config.COLOR_DAMAGE_ARROW
        ls = '--'
        lw = 2.2
    elif action == 'walk':
        arrow_color = config.COLOR_WALK_ARROW
        ls = '-'
        lw = 2.0
    elif action == 'jump':
        arrow_color = config.COLOR_JUMP_ARROW
        ls = '-'
        lw = 2.0
    else:
        arrow_color = 'orange'
        ls = ':'
        lw = 1.5

    if action == 'walk':
        ax.annotate('', xy=(c2, r2), xytext=(c1, r1),
                    arrowprops=dict(arrowstyle='->', color=arrow_color,
                                    lw=lw, ls=ls, alpha=alpha,
                                    connectionstyle='arc3,rad=0.0'))

    elif action == 'jump':
        ax.annotate('', xy=(c2, r2), xytext=(c1, r1),
                    arrowprops=dict(arrowstyle='->', color=arrow_color,
                                    lw=lw, ls=ls, alpha=alpha,
                                    connectionstyle='arc3,rad=0.4'))

    elif action == 'wall_bounce':
        if (r1, c1) != (r2, c2):
            mid_c = c1 + (c2 - c1) * 0.4
            mid_r = r1 + (r2 - r1) * 0.4
            ax.annotate('', xy=(mid_c, mid_r), xytext=(c1, r1),
                        arrowprops=dict(arrowstyle='->', color='orange',
                                        lw=1.5, ls=':', alpha=alpha * 0.9,
                                        connectionstyle='arc3,rad=0.0'))
            ax.text((c1 + mid_c) / 2, (r1 + mid_r) / 2 - 0.18, 'B',
                    ha='center', va='center', fontsize=6, color='orange',
                    fontweight='bold', alpha=alpha,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='orange',
                              alpha=0.75, lw=0.8))

    elif action == 'scan':
        ax.plot([c1, c2], [r1, r2], color='orange', lw=1, ls=':', alpha=0.5)

    if damage:
        ax.text(c2, r2 + 0.3, 'X', ha='center', va='center',
                fontsize=9, color='red', fontweight='bold', alpha=alpha)


def _hud_text(fig, step: Step, step_idx: int, total: int):
    """Render HUD overlay on the figure."""
    hud = (f"Step {step_idx}/{total}   "
           f"[Lives: {step.lives_after}]   "
           f"[Turns: {step.turns_after}]   "
           f"[Time: {step.time_after}]   "
           f"[Score: {step.score:.2f}  lower=better]")
    fig.suptitle(hud, fontsize=10, fontweight='bold',
                 y=0.98, color='#111111',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#FFFDE7', ec='#CCAA00', alpha=0.9))


def plot_ground_truth(grid, path, anchors, out_path):
    """Save ground-truth trajectory plot as PNG."""
    M = len(grid)
    N = len(grid[0])
    path_set = set(path)

    fig, ax = plt.subplots(figsize=(N * 0.9 + 1, M * 0.9 + 1))
    _draw_grid_cells(ax, grid, path_set=path_set)
    _setup_ax(ax, M, N, 'Ground-Truth Trajectory')

    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        ax.annotate('', xy=(c2, r2), xytext=(c1, r1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    a1, a2 = anchors
    for a, label in [(a1, 'A1'), (a2, 'A2')]:
        ax.text(a[1], a[0], label, ha='center', va='center',
                fontsize=8, color='black', fontweight='bold',
                bbox=dict(boxstyle='circle', fc='white', ec='black', alpha=0.8))

    legend_patches = [
        mpatches.Patch(color=config.COLOR_START,   label='Start (S)'),
        mpatches.Patch(color=config.COLOR_GOAL,    label='Goal (G)'),
        mpatches.Patch(color=config.COLOR_PATH,    label='Path (Land)'),
        mpatches.Patch(color=config.COLOR_VOLCANO, label='Volcano (V)'),
        mpatches.Patch(color=config.COLOR_WATER,   label='Water (W)'),
        mpatches.Patch(color=config.COLOR_WALL,    label='Wall (B)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=7, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Ground-truth trajectory saved -> {out_path}")


def plot_initial_grid(grid, out_path):
    """Save initial grid as PNG."""
    M = len(grid)
    N = len(grid[0])

    fig, ax = plt.subplots(figsize=(N * 0.9 + 1, M * 0.9 + 1))
    _draw_grid_cells(ax, grid)
    _setup_ax(ax, M, N, 'Initial Grid')

    legend_patches = [
        mpatches.Patch(color=config.COLOR_START,   label='Start (S)'),
        mpatches.Patch(color=config.COLOR_GOAL,    label='Goal (G)'),
        mpatches.Patch(color=config.COLOR_LAND,    label='Land (L)'),
        mpatches.Patch(color=config.COLOR_VOLCANO, label='Volcano (V)'),
        mpatches.Patch(color=config.COLOR_WATER,   label='Water (W)'),
        mpatches.Patch(color=config.COLOR_WALL,    label='Wall (B)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=7, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Initial grid saved -> {out_path}")


def animate_ex1(grid, steps, out_dir):
    """
    Render and save the Exercise-1 simulation.
    Each frame shows the current state with all arrows drawn so far.
    """
    M = len(grid)
    N = len(grid[0])
    frame_dir = os.path.join(out_dir, config.FRAME_DIR)
    os.makedirs(frame_dir, exist_ok=True)

    frame_paths = []
    damage_cells = set()
    agent_pos = (0, 0)

    for step_idx, step in enumerate(steps):
        if step.action in ('walk', 'jump'):
            agent_pos = step.to_pos
        elif step.action == 'wall_bounce':
            agent_pos = step.from_pos

        fig, ax = plt.subplots(figsize=(N + 2, M + 1.5))
        _draw_grid_cells(ax, grid)
        _setup_ax(ax, M, N, 'Oracle Agent — Exercise 1')

        for s in steps[:step_idx + 1]:
            if s.action in ('walk', 'jump', 'wall_bounce'):
                alpha = 1.0 if s == step else 0.45
                _draw_arrow(ax, s.from_pos, s.to_pos, s.action, s.damage, alpha=alpha)

        if step.damage:
            damage_cells.add(step.to_pos)
        for dc in damage_cells:
            ax.text(dc[1], dc[0], 'X', ha='center', va='center',
                    fontsize=14, color='red', fontweight='bold')

        r, c = agent_pos
        ax.plot(c, r, 'o', markersize=14, color=config.COLOR_AGENT,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)

        t_str = 'T:T' if step.sensor_thermal else 'T:F'
        s_str = 'S:T' if step.sensor_seismic else 'S:F'
        ax.text(c, r, f'{t_str}\n{s_str}', ha='center', va='center',
                fontsize=5, color='white', fontweight='bold', zorder=6)

        _hud_text(fig, step, step_idx + 1, len(steps))

        legend_patches = [
            mpatches.Patch(color='black',         label='Walk arrow'),
            mpatches.Patch(color='purple',        label='Jump arrow'),
            mpatches.Patch(color='red',           label='Damage'),
            mpatches.Patch(color=config.COLOR_LAND,    label='Land'),
            mpatches.Patch(color=config.COLOR_VOLCANO, label='Volcano'),
            mpatches.Patch(color=config.COLOR_WATER,   label='Water'),
            mpatches.Patch(color=config.COLOR_WALL,    label='Wall'),
        ]
        ax.legend(handles=legend_patches, loc='lower right',
                  fontsize=6, framealpha=0.9, ncol=2)

        plt.tight_layout()
        frame_path = os.path.join(frame_dir, f'frame_{step_idx:04d}.png')
        plt.savefig(frame_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)

    print(f"{len(frame_paths)} frames saved to {frame_dir}")
    _compile_gif(frame_paths, os.path.join(out_dir, config.SIMULATION_GIF))
    _compile_mp4(frame_dir, os.path.join(out_dir, config.SIMULATION_MP4))


def animate_ex2(grid, steps, out_dir):
    """
    Render and save the Exercise-2 dual-panel simulation.
    Left  -> Belief map (agent's perspective)
    Right -> Ground truth
    """
    M = len(grid)
    N = len(grid[0])
    frame_dir = os.path.join(out_dir, config.FRAME_DIR)
    os.makedirs(frame_dir, exist_ok=True)

    frame_paths  = []
    damage_cells = set()
    scan_cells   = set()
    scan_sensor_log: dict = {}

    agent_pos = (0, 0)

    for step_idx, step in enumerate(steps):
        fig, (ax_belief, ax_truth) = plt.subplots(
            1, 2, figsize=(N * 2 + 4, M + 2))

        if step.action in ('walk', 'jump', 'wall_bounce'):
            agent_pos = step.to_pos

        if step.belief_snapshot:
            _draw_belief_cells(ax_belief, step.belief_snapshot,
                               scan_cells=scan_cells,
                               damage_cells=damage_cells)
        else:
            _draw_grid_cells(ax_belief, grid)

        if step.scan_cell:
            scan_cells.add(step.scan_cell)
            if step.sensor_thermal is not None:
                scan_sensor_log[step.scan_cell] = (step.sensor_thermal, step.sensor_seismic)

        for (sr, sc), (t_val, s_val) in scan_sensor_log.items():
            t_lbl = 'T:T' if t_val else 'T:F'
            s_lbl = 'S:T' if s_val else 'S:F'
            ax_belief.text(sc, sr - 0.38, f'{t_lbl}', ha='center', va='center',
                           fontsize=4.5, color='black', fontweight='bold',
                           bbox=dict(fc='yellow', ec='none', alpha=0.7, pad=0.5))
            ax_belief.text(sc, sr + 0.38, f'{s_lbl}', ha='center', va='center',
                           fontsize=4.5, color='black', fontweight='bold',
                           bbox=dict(fc='yellow', ec='none', alpha=0.7, pad=0.5))

        for s in steps[:step_idx + 1]:
            if s.action in ('walk', 'jump', 'wall_bounce'):
                alpha = 1.0 if s == step else 0.4
                _draw_arrow(ax_belief, s.from_pos, s.to_pos,
                            s.action, s.damage, alpha=alpha)

        ar, ac = agent_pos
        ax_belief.plot(ac, ar, 'o', markersize=13, color='#1E1E1E',
                       markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax_belief.text(ac, ar, 'A', ha='center', va='center',
                       fontsize=7, color='white', fontweight='bold', zorder=6)

        if step.damage:
            damage_cells.add(step.to_pos)

        _setup_ax(ax_belief, M, N, 'Belief Map (Agent Perspective)')

        _draw_grid_cells(ax_truth, grid)

        for s in steps[:step_idx + 1]:
            if s.action in ('walk', 'jump', 'wall_bounce'):
                alpha = 1.0 if s == step else 0.4
                _draw_arrow(ax_truth, s.from_pos, s.to_pos,
                            s.action, s.damage, alpha=alpha)

        for dc in damage_cells:
            ax_truth.text(dc[1], dc[0], 'X', ha='center', va='center',
                          fontsize=14, color='red', fontweight='bold')

        ax_truth.plot(ac, ar, 'o', markersize=13, color='#1E1E1E',
                      markeredgecolor='white', markeredgewidth=1.5, zorder=5)

        _setup_ax(ax_truth, M, N, 'Ground Truth')

        _hud_text(fig, step, step_idx + 1, len(steps))
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        frame_path = os.path.join(frame_dir, f'frame_{step_idx:04d}.png')
        plt.savefig(frame_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)

    print(f"{len(frame_paths)} Ex2 frames saved to {frame_dir}")
    _compile_gif(frame_paths, os.path.join(out_dir, config.SIMULATION_GIF))
    _compile_mp4(frame_dir, os.path.join(out_dir, config.SIMULATION_MP4))


def _compile_gif(frame_paths, out_path):
    if not HAS_IMAGEIO:
        print("imageio not installed — skipping GIF compilation.")
        return
    if not frame_paths:
        return
    images = [imageio.imread(f) for f in frame_paths]
    duration = config.GIF_FRAME_DURATION_MS / 1000.0
    imageio.mimsave(out_path, images, duration=duration, loop=0)
    print(f"GIF saved -> {out_path}")


def _compile_mp4(frame_dir, out_path):
    """Try to compile frames to MP4 using ffmpeg."""
    import subprocess
    pattern = os.path.join(frame_dir, 'frame_%04d.png')
    fps = max(1, int(1000 / config.GIF_FRAME_DURATION_MS))
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', pattern,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        out_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0:
            print(f"MP4 saved -> {out_path}")
        else:
            print(f"ffmpeg failed: {result.stderr.decode()[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"MP4 compilation skipped ({e})")