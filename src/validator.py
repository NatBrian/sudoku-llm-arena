def validate_move(grid, row, col, value, box_width, box_height):
    size = len(grid)
    errors = []

    if not (0 <= row < size and 0 <= col < size):
        errors.append(f"Cell R{row+1}C{col+1} is out of bounds")
        return errors

    if not (1 <= value <= size):
        errors.append(f"Value {value} out of range 1-{size}")
        return errors

    if grid[row][col] != 0:
        errors.append(
            f"Cell R{row+1}C{col+1} already has value {grid[row][col]}"
        )

    if value in grid[row]:
        col_idx = grid[row].index(value) + 1
        errors.append(f"Row {row+1} already has {value} at C{col_idx}")

    for r in range(size):
        if grid[r][col] == value:
            errors.append(f"Column {col+1} already has {value} at R{r+1}")
            break

    box_start_row = (row // box_height) * box_height
    box_start_col = (col // box_width) * box_width
    for r in range(box_start_row, box_start_row + box_height):
        for c in range(box_start_col, box_start_col + box_width):
            if grid[r][c] == value:
                errors.append(
                    f"Box ({box_start_row+1}-{box_start_row+box_height}, "
                    f"{box_start_col+1}-{box_start_col+box_width}) "
                    f"already has {value} at R{r+1}C{c+1}"
                )
                return errors

    return errors


def is_solved(grid):
    size = len(grid)
    for row in range(size):
        for col in range(size):
            if grid[row][col] == 0:
                return False
    return True


def is_grid_stuck(grid, box_width, box_height):
    size = len(grid)
    for row in range(size):
        for col in range(size):
            if grid[row][col] == 0:
                for val in range(1, size + 1):
                    errs = validate_move(
                        grid, row, col, val, box_width, box_height
                    )
                    if not errs:
                        return False
    return True


def count_empty(grid):
    return sum(1 for row in grid for c in row if c == 0)


def candidates_for(grid, box_width, box_height, row, col):
    size = len(grid)
    return [
        val for val in range(1, size + 1)
        if not validate_move(grid, row, col, val, box_width, box_height)
    ]


def compute_candidates(grid, box_width, box_height):
    size = len(grid)
    return {
        (r, c): set(candidates_for(grid, box_width, box_height, r, c))
        for r in range(size) for c in range(size) if grid[r][c] == 0
    }


def find_naked_singles(grid, box_width, box_height):
    size = len(grid)
    results = []
    for row in range(size):
        for col in range(size):
            if grid[row][col] == 0:
                candidates = candidates_for(grid, box_width, box_height, row, col)
                if len(candidates) == 1:
                    results.append((row, col, candidates[0]))
    return results


def _row_cells(row, size):
    return [(row, c) for c in range(size)]


def _col_cells(col, size):
    return [(r, col) for r in range(size)]


def _box_cells(row, col, box_width, box_height):
    box_start_row = (row // box_height) * box_height
    box_start_col = (col // box_width) * box_width
    return [
        (r, c)
        for r in range(box_start_row, box_start_row + box_height)
        for c in range(box_start_col, box_start_col + box_width)
    ]


def _all_boxes(box_width, box_height, size):
    return [
        _box_cells(box_start_row, box_start_col, box_width, box_height)
        for box_start_row in range(0, size, box_height)
        for box_start_col in range(0, size, box_width)
    ]


def _all_units(box_width, box_height):
    size = box_width * box_height
    units = [_row_cells(r, size) for r in range(size)]
    units += [_col_cells(c, size) for c in range(size)]
    units += _all_boxes(box_width, box_height, size)
    return units


def find_hidden_singles(grid, box_width, box_height):
    """Digits that can only go in one cell within some row/col/box, even if
    that cell still has other raw candidates (unlike a naked single)."""
    candidates = compute_candidates(grid, box_width, box_height)
    size = len(grid)
    seen = set()
    results = []
    for unit in _all_units(box_width, box_height):
        empty_cells = [cell for cell in unit if cell in candidates]
        for value in range(1, size + 1):
            holders = [cell for cell in empty_cells if value in candidates[cell]]
            if len(holders) == 1 and holders[0] not in seen:
                r, c = holders[0]
                if len(candidates[(r, c)]) > 1:  # not already a naked single
                    seen.add((r, c))
                    results.append((r, c, value))
    return results


def _is_hidden_single(candidates, box_width, box_height, row, col, value):
    size = box_width * box_height
    for unit in (
        _row_cells(row, size),
        _col_cells(col, size),
        _box_cells(row, col, box_width, box_height),
    ):
        holders = [cell for cell in unit if cell in candidates and value in candidates[cell]]
        if len(holders) == 1 and holders[0] == (row, col):
            return True
    return False


def eliminate_naked_pairs(candidates, box_width, box_height):
    """Two cells sharing a unit with the exact same 2 candidates -> those 2
    values can't appear anywhere else in that unit. Mutates `candidates`,
    returns a list of (cell, {eliminated values}, reason) events."""
    eliminated = []
    for unit in _all_units(box_width, box_height):
        cells = [c for c in unit if c in candidates]
        pairs = [c for c in cells if len(candidates[c]) == 2]
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                a, b = pairs[i], pairs[j]
                if candidates[a] != candidates[b]:
                    continue
                values = candidates[a]
                for cell in cells:
                    if cell in (a, b):
                        continue
                    removed = candidates[cell] & values
                    if removed:
                        candidates[cell] -= values
                        eliminated.append((
                            cell, removed,
                            f"naked pair {sorted(values)} at R{a[0]+1}C{a[1]+1}/R{b[0]+1}C{b[1]+1}",
                        ))
    return eliminated


def eliminate_pointing_pairs(candidates, box_width, box_height):
    """If a digit's candidates within a box all share one row (or column),
    it can be eliminated from the rest of that row/column outside the box."""
    size = box_width * box_height
    eliminated = []
    for box in _all_boxes(box_width, box_height, size):
        box_set = set(box)
        box_cells = [c for c in box if c in candidates]
        for value in range(1, size + 1):
            holders = [c for c in box_cells if value in candidates[c]]
            if len(holders) < 2:
                continue
            rows = {r for r, _ in holders}
            cols = {c for _, c in holders}
            if len(rows) == 1:
                row = next(iter(rows))
                for cell in _row_cells(row, size):
                    if cell not in box_set and cell in candidates and value in candidates[cell]:
                        candidates[cell].discard(value)
                        eliminated.append((cell, {value}, f"pointing pair: {value} confined to row {row+1} within a box"))
            if len(cols) == 1:
                col = next(iter(cols))
                for cell in _col_cells(col, size):
                    if cell not in box_set and cell in candidates and value in candidates[cell]:
                        candidates[cell].discard(value)
                        eliminated.append((cell, {value}, f"pointing pair: {value} confined to column {col+1} within a box"))
    return eliminated


def eliminate_box_line_reduction(candidates, box_width, box_height):
    """Converse of pointing pairs: if a digit's candidates within a row (or
    column) all fall in one box, it can be eliminated from the rest of that
    box outside the row/column."""
    size = box_width * box_height
    eliminated = []
    lines = [_row_cells(r, size) for r in range(size)] + [_col_cells(c, size) for c in range(size)]
    for line in lines:
        line_set = set(line)
        line_cells = [c for c in line if c in candidates]
        for value in range(1, size + 1):
            holders = [c for c in line_cells if value in candidates[c]]
            if len(holders) < 2:
                continue
            box_ids = {(r // box_height, c // box_width) for r, c in holders}
            if len(box_ids) == 1:
                box_cells = _box_cells(holders[0][0], holders[0][1], box_width, box_height)
                for cell in box_cells:
                    if cell not in line_set and cell in candidates and value in candidates[cell]:
                        candidates[cell].discard(value)
                        eliminated.append((cell, {value}, f"box-line reduction: {value} confined to one box within a line"))
    return eliminated


def classify_move(grid, box_width, box_height, row, col, value):
    """Returns (technique_name, justification) if placing `value` at
    (row, col) is justified by an implemented solving technique, else None.
    Tries naked/hidden singles first, then applies elimination techniques
    (naked pairs, pointing pairs, box-line reduction) to a fixpoint and
    re-checks — these don't place digits themselves but narrow candidates
    until a naked/hidden single appears."""
    if grid[row][col] != 0:
        return None
    candidates = compute_candidates(grid, box_width, box_height)
    if (row, col) not in candidates or value not in candidates[(row, col)]:
        return None

    if len(candidates[(row, col)]) == 1:
        return (
            "naked_single",
            f"Row {row+1}, column {col+1}, and its box already rule out every "
            f"digit except {value} — a naked single.",
        )
    if _is_hidden_single(candidates, box_width, box_height, row, col, value):
        return (
            "hidden_single",
            f"{value} can only go in R{row+1}C{col+1} within its row, column, "
            f"or box, even though that cell has other candidates — a hidden single.",
        )

    applied = []
    for _ in range(6):
        changed = False
        for elim_fn, label in (
            (eliminate_naked_pairs, "naked pair"),
            (eliminate_pointing_pairs, "pointing pair"),
            (eliminate_box_line_reduction, "box-line reduction"),
        ):
            if elim_fn(candidates, box_width, box_height):
                changed = True
                applied.append(label)
        if not changed:
            break
        techniques = ", ".join(dict.fromkeys(applied))
        if len(candidates[(row, col)]) == 1 and value in candidates[(row, col)]:
            return (
                "naked_single_after_elimination",
                f"After eliminating candidates via {techniques}, R{row+1}C{col+1} "
                f"narrows to {value}.",
            )
        if _is_hidden_single(candidates, box_width, box_height, row, col, value):
            return (
                "hidden_single_after_elimination",
                f"After eliminating candidates via {techniques}, {value} is "
                f"confined to R{row+1}C{col+1} within a unit.",
            )
    return None


def remaining_candidates(grid, box_width, box_height):
    """Total candidate-count across all empty cells — a proxy for how far a
    grid still is from being solved, used to reward-shape GRPO moves that
    tighten constraints."""
    candidates = compute_candidates(grid, box_width, box_height)
    return sum(len(v) for v in candidates.values())


def grids_equal(a, b):
    if len(a) != len(b):
        return False
    for r in range(len(a)):
        if len(a[r]) != len(b[r]):
            return False
        for c in range(len(a[r])):
            if a[r][c] != b[r][c]:
                return False
    return True
