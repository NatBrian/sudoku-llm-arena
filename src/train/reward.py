"""GRPO reward function (tier2/tier3). Reuses the exact same parser/validator
the real eval harness (loop.py) uses, so a policy optimized here is optimizing
for literally the same notion of "valid move" it'll be judged on later.
"""
import json
from copy import deepcopy

from ..parser import parse_response
from ..validator import validate_move, classify_move, remaining_candidates, candidates_for

VALID_MOVE_REWARD = 1.0
SOLUTION_MATCH_BONUS = 1.0
TECHNIQUE_BONUS = 0.5
PROGRESS_BONUS_SCALE = 0.05  # per candidate eliminated by the move
PROGRESS_BONUS_CAP = 0.5
INVALID_MOVE_REWARD = -1.0
STUCK_REWARD = -1.0
UNPARSEABLE_REWARD = -1.0
# Backtracking a genuine mistake is as valuable as a correct forward move —
# it's the only tool the eval harness (loop.py) gives the model to escape a
# self-created dead end, and it was previously falling through to
# UNPARSEABLE_REWARD, training the model to never use it.
BACKTRACK_CORRECT_REWARD = 1.0
BACKTRACK_WRONG_REWARD = -0.3


def _completion_text(completion):
    if isinstance(completion, str):
        return completion
    # trl passes conversational completions as a list of chat messages
    return completion[-1]["content"]


def reward_func(prompts, completions, grid_before, box_width, box_height, size, solution, clues, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        grid = json.loads(grid_before[i])
        bw, bh, sz = box_width[i], box_height[i], size[i]
        sol = json.loads(solution[i])
        clue_grid = json.loads(clues[i])

        parsed = parse_response(text, sz)

        if parsed["stuck"]:
            rewards.append(STUCK_REWARD)
            continue

        if parsed["backtrack"]:
            # Mirrors loop.py's own accept/refuse logic: clue cells and
            # already-empty cells refuse the backtrack; otherwise it clears
            # the cell — reward that clear based on whether it undid an
            # actual mistake (vs. solution) or a cell that was already right.
            bt = parsed["backtrack"]
            r, c = bt["row"], bt["col"]
            if clue_grid[r][c] != 0 or grid[r][c] == 0:
                rewards.append(INVALID_MOVE_REWARD)
            elif grid[r][c] != sol[r][c]:
                rewards.append(BACKTRACK_CORRECT_REWARD)
            else:
                rewards.append(BACKTRACK_WRONG_REWARD)
            continue

        move = parsed["move"]
        if not move:
            rewards.append(UNPARSEABLE_REWARD)
            continue

        errors = validate_move(grid, move["row"], move["col"], move["value"], bw, bh)
        if errors:
            rewards.append(INVALID_MOVE_REWARD)
            continue

        reward = VALID_MOVE_REWARD
        if sol[move["row"]][move["col"]] == move["value"]:
            # legal-but-wrong-for-this-solution placements can still dead-end
            # the puzzle later, so reward matching the unique solution more.
            reward += SOLUTION_MATCH_BONUS

        # Denser signal on top of the sparse valid/solution-match reward:
        # reward moves the model can actually justify via a known technique,
        # and moves that tighten the puzzle's remaining candidate space, not
        # just any legal-but-uninformative placement.
        if classify_move(grid, bw, bh, move["row"], move["col"], move["value"]):
            reward += TECHNIQUE_BONUS

        # Compare candidate counts over the OTHER (still-empty) cells only —
        # excluding the placed cell's own candidates, which always disappear
        # on any move and would otherwise swamp the signal — so this isolates
        # how much this specific move's constraints propagate elsewhere.
        own_candidates = len(candidates_for(grid, bw, bh, move["row"], move["col"]))
        before_others = remaining_candidates(grid, bw, bh) - own_candidates
        grid_after = deepcopy(grid)
        grid_after[move["row"]][move["col"]] = move["value"]
        after_others = remaining_candidates(grid_after, bw, bh)
        progress = min(PROGRESS_BONUS_CAP, PROGRESS_BONUS_SCALE * max(0, before_others - after_others))
        reward += progress

        rewards.append(reward)
    return rewards
