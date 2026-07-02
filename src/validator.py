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


def find_naked_singles(grid, box_width, box_height):
    size = len(grid)
    results = []
    for row in range(size):
        for col in range(size):
            if grid[row][col] == 0:
                candidates = []
                for val in range(1, size + 1):
                    errs = validate_move(
                        grid, row, col, val, box_width, box_height
                    )
                    if not errs:
                        candidates.append(val)
                if len(candidates) == 1:
                    results.append((row, col, candidates[0]))
    return results


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
