import re

MOVE_PATTERN = re.compile(r"MOVE:\s*R(\d+)\s*C(\d+)\s*[=:]\s*(\d+)", re.IGNORECASE)
GRID_ROW_PATTERN = re.compile(r"\[\s*([\d,\s]+)\s*\]")


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
    stuck = is_stuck_response(text)

    return {
        "move": move,
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


def detect_full_grid(text, size):
    lines = text.strip().split("\n")
    numbers = []
    for line in lines:
        nums = []
        for token in re.findall(r"\b[1-9]\b", line):
            nums.append(int(token))
        if len(nums) == size:
            numbers.append(nums)
    if len(numbers) == size:
        return numbers
    return None
