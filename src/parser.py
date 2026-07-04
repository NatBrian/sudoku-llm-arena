import json
import re

MOVE_PATTERN = re.compile(r"MOVE:\s*R(\d+)\s*C(\d+)\s*[=:]\s*(\d+)", re.IGNORECASE)
BACKTRACK_PATTERN = re.compile(r"BACKTRACK\s+at\s+R(\d+)\s*C(\d+)", re.IGNORECASE)


def parse_response(text, size):
    move_match = MOVE_PATTERN.search(text)

    move = None
    move_raw = ""
    reasoning = ""

    if move_match:
        row = int(move_match.group(1)) - 1
        col = int(move_match.group(2)) - 1
        value = int(move_match.group(3))
        move = {"row": row, "col": col, "value": value}
        move_raw = move_match.group(0)
        reasoning = text[: move_match.start()].strip()
    else:
        reasoning = text.strip()

    reasoning = re.sub(r"^REASONING:\s*", "", reasoning, flags=re.IGNORECASE).strip()

    backtrack = None
    if not move:
        bt_match = BACKTRACK_PATTERN.search(text)
        if bt_match:
            backtrack = {
                "row": int(bt_match.group(1)) - 1,
                "col": int(bt_match.group(2)) - 1,
            }

    # Only treat a response as "stuck" if it didn't also give us a move or a
    # backtrack to act on — models often reason about what *doesn't* work
    # ("no possible value here...") before landing on the actual answer, and
    # that reasoning text alone shouldn't discard a valid move.
    stuck = is_stuck_response(text) and not move and not backtrack

    return {
        "move": move,
        "backtrack": backtrack,
        "reasoning": reasoning,
        "move_raw": move_raw,
        "stuck": stuck,
        "raw": text,
    }


def is_stuck_response(text):
    upper = text.upper()
    triggers = [
        "I'M STUCK",
        "I AM STUCK",
        "CANNOT FIND",
        "CAN'T FIND",
        "NO POSSIBLE",
        "NO VALID",
        "UNABLE TO",
    ]
    return any(t in upper for t in triggers)


def parse_full_grid(text, size):
    """Parse a single-shot full-solution response: a JSON grid if present,
    otherwise the last `size` lines that each contain exactly `size` digits
    1-size (models often echo the puzzle before the final answer)."""
    json_match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
    if json_match:
        try:
            grid = json.loads(json_match.group(0))
            if len(grid) == size and all(len(row) == size for row in grid):
                return [[int(v) for v in row] for row in grid]
        except (ValueError, TypeError):
            pass

    lines = text.strip().split("\n")
    candidate_blocks = []
    block = []
    for line in lines:
        nums = [int(t) for t in re.findall(r"\b[1-9]\b", line)]
        if len(nums) == size:
            block.append(nums)
        elif block:
            candidate_blocks.append(block)
            block = []
    if block:
        candidate_blocks.append(block)

    for block in reversed(candidate_blocks):
        if len(block) == size:
            return block
    return None
