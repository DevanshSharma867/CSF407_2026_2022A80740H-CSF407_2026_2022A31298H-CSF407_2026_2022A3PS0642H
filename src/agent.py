"""
agent.py
Oracle Agent — CS F407 AI Assignment

Two agent modes:
  Exercise 1: OracleAgent  — Deterministic, full grid visibility
  Exercise 2: ProbOracle   — Probabilistic, Bayesian belief + noisy sensors

Score to MINIMISE: (turn_units + time_units) / lives_remaining
"""

import heapq
import random
import config


ORTHOGONAL_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

JUMP_WALL_BELIEF_THRESH = 0.85
LAND_WALL_BELIEF_THRESH = 0.85


def _in_bounds(r, c):
    return 0 <= r < config.GRID_ROWS and 0 <= c < config.GRID_COLS


def _is_hazard(cell_type):
    return cell_type in (config.CELL_VOLCANO, config.CELL_WATER)


def _is_wall(cell_type):
    return cell_type == config.CELL_WALL


def _manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


class Step:
    def __init__(self, action, from_pos, to_pos, damage,
                 lives_after, turns_after, time_after,
                 sensor_thermal=None, sensor_seismic=None,
                 belief_snapshot=None, scan_cell=None):
        self.action = action
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.damage = damage
        self.lives_after = lives_after
        self.turns_after = turns_after
        self.time_after = time_after
        self.score = (turns_after + time_after) / max(lives_after, 1)
        self.sensor_thermal = sensor_thermal
        self.sensor_seismic = sensor_seismic
        self.belief_snapshot = belief_snapshot
        self.scan_cell = scan_cell

    def __repr__(self):
        return (f"Step({self.action}: {self.from_pos}->{self.to_pos} "
                f"dmg={self.damage} lives={self.lives_after} "
                f"T={self.turns_after} t={self.time_after} score={self.score:.2f})")


class OracleAgent:
    """
    Deterministic agent with full grid visibility.
    """

    def __init__(self, grid: list):
        self.grid = grid
        self.start = (0, 0)
        self.goal = (config.GRID_ROWS - 1, config.GRID_COLS - 1)

    def sense(self, r, c):
        thermal = seismic = False
        for dr, dc in ORTHOGONAL_DELTAS:
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc):
                ct = self.grid[nr][nc]
                if ct == config.CELL_VOLCANO:
                    thermal = True
                if ct == config.CELL_WATER:
                    seismic = True
        return thermal, seismic

    def _h(self, r, c):
        dist = _manhattan(r, c, self.goal[0], self.goal[1])
        return dist * config.WALK_TIME_COST

    def _life_penalty(self, cell_type):
        return 1 if _is_hazard(cell_type) else 0

    def search(self):
        goal_r, goal_c = self.goal
        sr, sc = self.start
        lives0 = config.AGENT_START_LIVES

        h0 = self._h(sr, sc)
        f0 = h0 / max(lives0, 1)

        heap = [(f0, 0, 0, lives0, sr, sc, [(sr, sc, 'start', False)])]
        visited = {}

        while heap:
            f, turns, time, lives, r, c, history = heapq.heappop(heap)

            if lives <= 0:
                continue

            state = (r, c, lives)
            g_now = turns + time
            if state in visited and visited[state] <= g_now:
                continue
            visited[state] = g_now

            if (r, c) == (goal_r, goal_c):
                return self._build_steps(history)

            for dr, dc in ORTHOGONAL_DELTAS:
                # WALK
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc):
                    ct = self.grid[nr][nc]
                    if not _is_wall(ct):
                        nt = turns + config.TURN_COST
                        nm = time + config.WALK_TIME_COST
                        nl = max(lives - self._life_penalty(ct), 0)
                        g2 = nt + nm
                        f2 = (g2 + self._h(nr, nc)) / max(nl, 1)
                        heapq.heappush(heap, (
                            f2, nt, nm, nl, nr, nc,
                            history + [(nr, nc, 'walk', _is_hazard(ct))]
                        ))

                # JUMP
                jr, jc = r + 2 * dr, c + 2 * dc
                mr, mc = r + dr, c + dc
                if _in_bounds(jr, jc) and _in_bounds(mr, mc):
                    mid_ct = self.grid[mr][mc]
                    land_ct = self.grid[jr][jc]
                    if not _is_wall(mid_ct) and not _is_wall(land_ct):
                        nt = turns + config.TURN_COST
                        nm = time + config.JUMP_TIME_COST
                        nl = max(lives - self._life_penalty(land_ct), 0)
                        g2 = nt + nm
                        f2 = (g2 + self._h(jr, jc)) / max(nl, 1)
                        heapq.heappush(heap, (
                            f2, nt, nm, nl, jr, jc,
                            history + [(jr, jc, 'jump', _is_hazard(land_ct))]
                        ))

        return []

    def _build_steps(self, history):
        steps = []
        lives = config.AGENT_START_LIVES
        turns = 0
        time = 0
        agent_r, agent_c = history[0][0], history[0][1]

        for i, (r, c, action, dmg) in enumerate(history):
            if action == 'start':
                continue

            prev_r, prev_c = history[i - 1][0], history[i - 1][1]

            if action == 'walk':
                turns += config.TURN_COST
                time += config.WALK_TIME_COST
                agent_r, agent_c = r, c
            elif action == 'jump':
                turns += config.TURN_COST
                time += config.JUMP_TIME_COST
                agent_r, agent_c = r, c

            if dmg:
                lives = max(lives - 1, 0)

            thermal, seismic = self.sense(agent_r, agent_c)

            steps.append(Step(
                action=action,
                from_pos=(prev_r, prev_c),
                to_pos=(r, c),
                damage=dmg,
                lives_after=lives,
                turns_after=turns,
                time_after=time,
                sensor_thermal=thermal,
                sensor_seismic=seismic,
            ))

        return steps


class BeliefCell:
    TYPES = [config.CELL_VOLCANO, config.CELL_WATER,
             config.CELL_LAND, config.CELL_WALL]

    def __init__(self):
        self.probs = {
            config.CELL_VOLCANO: config.PRIOR_VOLCANO,
            config.CELL_WATER: config.PRIOR_WATER,
            config.CELL_LAND: config.PRIOR_LAND,
            config.CELL_WALL: config.PRIOR_WALL,
        }
        self.scan_count = 0
        self.blocked = False
        self.revealed = False

    def bayesian_update(self, thermal: bool, seismic: bool):
        new_probs = {}
        for ct in self.TYPES:
            p_th, p_se = config.CPT[ct]
            lh_th = p_th if thermal else (1.0 - p_th)
            lh_se = p_se if seismic else (1.0 - p_se)
            new_probs[ct] = self.probs[ct] * lh_th * lh_se

        total = sum(new_probs.values())
        if total > 1e-12:
            self.probs = {ct: v / total for ct, v in new_probs.items()}

        self.scan_count += 1
        if self.scan_count >= config.MAX_SCANS_PER_CELL:
            self.blocked = True

    def set_known(self, true_type: str):
        self.probs = {ct: (1.0 if ct == true_type else 0.0)
                      for ct in self.TYPES}
        self.revealed = True

    @property
    def p_hazard(self):
        return self.probs[config.CELL_VOLCANO] + self.probs[config.CELL_WATER]

    @property
    def p_wall(self):
        return self.probs[config.CELL_WALL]

    def copy_probs(self):
        return dict(self.probs)


class ProbOracle:
    LIFE_HAZARD_WEIGHT = 6

    def __init__(self, grid: list):
        self.grid = grid
        self.start = (0, 0)
        self.goal = (config.GRID_ROWS - 1, config.GRID_COLS - 1)

        M, N = config.GRID_ROWS, config.GRID_COLS
        self.belief = [[BeliefCell() for _ in range(N)] for _ in range(M)]

        self.belief[0][0].set_known(config.CELL_LAND)
        goal_r, goal_c = self.goal
        self.belief[goal_r][goal_c].set_known(config.CELL_LAND)
        self.belief[goal_r][goal_c].revealed = True

    def _noisy_read(self, r, c) -> tuple:
        true_type = self.grid[r][c]
        if true_type in (config.CELL_START, config.CELL_GOAL):
            true_type = config.CELL_LAND
        if true_type not in config.CPT:
            true_type = config.CELL_LAND

        p_th, p_se = config.CPT[true_type]
        thermal = random.random() < p_th
        seismic = random.random() < p_se
        return thermal, seismic

    def scan(self, r, c) -> tuple:
        bc = self.belief[r][c]
        if bc.blocked or bc.revealed:
            return None, None
        thermal, seismic = self._noisy_read(r, c)
        bc.bayesian_update(thermal, seismic)
        return thermal, seismic

    def _snapshot(self) -> dict:
        M, N = config.GRID_ROWS, config.GRID_COLS
        return {(r, c): self.belief[r][c].copy_probs()
                for r in range(M) for c in range(N)}

    def _h(self, r, c) -> float:
        return _manhattan(r, c, self.goal[0], self.goal[1]) * config.WALK_TIME_COST

    def _edge_cost(self, r, c, action: str) -> float:
        bc = self.belief[r][c]
        if action == 'walk':
            base = config.TURN_COST + config.WALK_TIME_COST
        else:
            base = config.TURN_COST + config.JUMP_TIME_COST
        return base + bc.p_hazard * self.LIFE_HAZARD_WEIGHT

    def _plan(self, start_r, start_c, current_lives) -> list:
        """
        Belief-only A* with lives tracking in state key.
        """
        goal_r, goal_c = self.goal
        h0 = self._h(start_r, start_c)
        f0 = h0 / max(current_lives, 1)

        heap = [(f0, 0.0, current_lives, start_r, start_c, [(start_r, start_c, 'start')])]
        visited = {}

        while heap:
            f, g, cl, r, c, path = heapq.heappop(heap)

            if (r, c) == (goal_r, goal_c):
                return path

            state = (r, c, cl)
            if state in visited and visited[state] <= g:
                continue
            visited[state] = g

            for dr, dc in ORTHOGONAL_DELTAS:
                # WALK
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc):
                    bc = self.belief[nr][nc]
                    if bc.p_wall < LAND_WALL_BELIEF_THRESH:
                        eg = g + self._edge_cost(nr, nc, 'walk')
                        nl = max(cl - (1 if bc.p_hazard > 0.5 else 0), 1)
                        f2 = (eg + self._h(nr, nc)) / max(nl, 1)
                        heapq.heappush(heap, (
                            f2, eg, nl, nr, nc, path + [(nr, nc, 'walk')]
                        ))

                # JUMP
                jr, jc = r + 2 * dr, c + 2 * dc
                mr, mc = r + dr, c + dc
                if _in_bounds(jr, jc) and _in_bounds(mr, mc):
                    mid_bc = self.belief[mr][mc]
                    land_bc = self.belief[jr][jc]
                    if (mid_bc.p_wall < JUMP_WALL_BELIEF_THRESH and
                        land_bc.p_wall < LAND_WALL_BELIEF_THRESH):
                        eg = g + self._edge_cost(jr, jc, 'jump')
                        nl = max(cl - (1 if land_bc.p_hazard > 0.5 else 0), 1)
                        f2 = (eg + self._h(jr, jc)) / max(nl, 1)
                        heapq.heappush(heap, (
                            f2, eg, nl, jr, jc, path + [(jr, jc, 'jump')]
                        ))

        return []

    def _execute_step(self, from_r, from_c, to_r, to_c,
                      action: str, lives: int) -> tuple:
        if action == 'jump':
            dr = (to_r - from_r) // 2
            dc = (to_c - from_c) // 2
            mr, mc = from_r + dr, from_c + dc
            if _is_wall(self.grid[mr][mc]):
                self.belief[mr][mc].set_known(config.CELL_WALL)
                return (from_r, from_c, lives, config.WALL_TURN_COST, 0, False, True)

        if _is_wall(self.grid[to_r][to_c]):
            self.belief[to_r][to_c].set_known(config.CELL_WALL)
            return (from_r, from_c, lives, config.WALL_TURN_COST, 0, False, True)

        true_ct = self.grid[to_r][to_c]
        damage = _is_hazard(true_ct)
        new_lives = max(lives - (1 if damage else 0), 0)

        td, md = (config.TURN_COST, config.WALK_TIME_COST) if action == 'walk' else \
                 (config.TURN_COST, config.JUMP_TIME_COST)

        reveal = true_ct
        if reveal in (config.CELL_START, config.CELL_GOAL):
            reveal = config.CELL_LAND
        if reveal in BeliefCell.TYPES:
            self.belief[to_r][to_c].set_known(reveal)

        return (to_r, to_c, new_lives, td, md, damage, False)

    def search(self) -> list:
        steps, lives, turns, time = [], config.AGENT_START_LIVES, 0, 0
        r, c, goal_r, goal_c = self.start[0], self.start[1], self.goal[0], self.goal[1]

        MAX_ITERS = config.GRID_ROWS * config.GRID_COLS * 10

        for _ in range(MAX_ITERS):
            if lives <= 0 or (r, c) == (goal_r, goal_c):
                break

            plan = self._plan(r, c, lives)
            if len(plan) < 2:
                break
            next_r, next_c, next_action = plan[1]

            bc_next = self.belief[next_r][next_c]
            while (bc_next.p_hazard > config.RISK_SCAN_THRESHOLD and
                   not bc_next.blocked and not bc_next.revealed):
                thermal, seismic = self.scan(next_r, next_c)
                if thermal is not None:
                    turns += 1
                    steps.append(Step('scan', (r, c), (next_r, next_c), False, lives,
                                      turns, time, thermal, seismic, self._snapshot(), (next_r, next_c)))
                    plan = self._plan(r, c, lives)
                    if len(plan) < 2:
                        break
                    next_r, next_c, next_action = plan[1]
                    bc_next = self.belief[next_r][next_c]
                else:
                    break

            if len(plan) < 2:
                break
            ar, ac, new_lives, td, md, damage, bounced = self._execute_step(r, c, next_r, next_c, next_action, lives)
            turns += td
            time += md
            lives = new_lives

            if bounced:
                steps.append(Step('wall_bounce', (r, c), (next_r, next_c), False, lives, turns, time, belief_snapshot=self._snapshot()))
                continue

            thermal_now, seismic_now = self._noisy_read(ar, ac)
            steps.append(Step(next_action, (r, c), (ar, ac), damage, lives, turns, time, thermal_now, seismic_now, self._snapshot()))
            r, c = ar, ac

        return steps