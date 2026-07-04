import time
from copy import deepcopy
from datetime import datetime

import litellm
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    NotFoundError,
    PermissionDeniedError,
)

from . import config
from .validator import validate_move, is_solved, count_empty, grids_equal
from .parser import parse_response, parse_full_grid
from .strategies import build_prompt, build_single_shot_prompt, STRATEGIES
from .storage import save_raw_run

litellm.suppress_debug_info = True

# Retrying these just burns the retry budget on a call that will never
# succeed — wrong API key, bad model id, prompt too long, etc. Fail fast
# instead of waiting through 3 exponential-backoff sleeps for nothing.
NON_RETRYABLE_ERRORS = (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
)


def model_slug(model):
    return model.replace("/", "-")


class GameLoop:
    def __init__(self, puzzle, model, protocol, strategy_id=None, run_number=1):
        self.puzzle = puzzle
        self.model = model
        self.protocol = protocol  # "multi-step" | "single-shot"
        self.strategy_id = strategy_id
        self.run_number = run_number

        self.size = puzzle["size"]
        self.box_width = puzzle["box_width"]
        self.box_height = puzzle["box_height"]

        self.clues = _normalize_grid(puzzle["clues"], self.size)
        self.grid = deepcopy(self.clues)
        self.solution = _normalize_grid(puzzle["solution"], self.size)

        self.max_turns = count_empty(self.grid) * config.MAX_TURNS_MULTIPLIER
        self.turns = []
        self.consecutive_errors = 0
        self.parse_failures = 0

    def run(self, log_prefix=""):
        start_time = datetime.now()
        if self.protocol == "single-shot":
            self._run_single_shot(log_prefix, start_time)
        else:
            self._run_multi_step(log_prefix, start_time)
        result = self._build_result(start_time)
        save_raw_run(result)
        return result

    def _run_id(self):
        strat = self.strategy_id or self.protocol
        return f"{self.puzzle['puzzle_id']}_{model_slug(self.model)}_{strat}_r{self.run_number}"

    def _run_multi_step(self, log_prefix, start_time):
        pfx = log_prefix or f"[{self.model} {self.strategy_id} r{self.run_number}]"

        for turn in range(1, self.max_turns + 1):
            remaining = count_empty(self.grid)
            print(f"  {pfx} Turn {turn}/{self.max_turns} ({remaining} cells left)...", flush=True)
            prompt = build_prompt(self.strategy_id, self.puzzle, self.grid)

            api_response = self._call_api(prompt, pfx)
            if api_response is None:
                print(f"  {pfx} API failed at turn {turn}, giving up", flush=True)
                break

            parsed = parse_response(api_response["content"], self.size)

            turn_data = {
                "turn": turn,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "raw_response": api_response,
                "reasoning_raw": parsed["reasoning"],
                "move_raw": parsed["move_raw"],
                "parsed_move": parsed["move"],
                "stuck": parsed["stuck"],
                "valid": False,
                "validation_errors": [],
                "grid_before": deepcopy(self.grid),
                "grid_after": None,
            }

            if parsed["stuck"]:
                turn_data["validation_errors"] = ["LLM reported being stuck"]
                turn_data["grid_after"] = deepcopy(self.grid)
                self.turns.append(turn_data)
                print(f"    {pfx} LLM is stuck — ending run", flush=True)
                break

            if parsed["backtrack"]:
                bt = parsed["backtrack"]
                if self.clues[bt["row"]][bt["col"]] != 0:
                    turn_data["validation_errors"] = [
                        f"Refused BACKTRACK at R{bt['row']+1}C{bt['col']+1}: that's a clue cell"
                    ]
                    turn_data["grid_after"] = deepcopy(self.grid)
                    self.consecutive_errors += 1
                elif self.grid[bt["row"]][bt["col"]] == 0:
                    turn_data["validation_errors"] = [
                        f"Refused BACKTRACK at R{bt['row']+1}C{bt['col']+1}: cell already empty"
                    ]
                    turn_data["grid_after"] = deepcopy(self.grid)
                    self.consecutive_errors += 1
                else:
                    self.grid[bt["row"]][bt["col"]] = 0
                    turn_data["valid"] = True
                    turn_data["backtrack"] = bt
                    turn_data["grid_after"] = deepcopy(self.grid)
                    turn_data["validation_errors"] = []
                    self.consecutive_errors = 0
                    print(f"    {pfx} <- BACKTRACK R{bt['row']+1}C{bt['col']+1} cleared", flush=True)
                self.turns.append(turn_data)
                if self.consecutive_errors >= config.MAX_CONSECUTIVE_ERRORS:
                    print(f"    {pfx} Too many consecutive errors ({self.consecutive_errors}) — stopping", flush=True)
                    break
                continue

            if parsed["move"]:
                mv = parsed["move"]
                errors = validate_move(
                    self.grid, mv["row"], mv["col"], mv["value"],
                    self.box_width, self.box_height,
                )
                if not errors:
                    self.grid[mv["row"]][mv["col"]] = mv["value"]
                    turn_data["valid"] = True
                    turn_data["grid_after"] = deepcopy(self.grid)
                    turn_data["validation_errors"] = []
                    self.consecutive_errors = 0
                    print(f"    {pfx} -> R{mv['row']+1}C{mv['col']+1} = {mv['value']} [valid]", flush=True)
                else:
                    turn_data["validation_errors"] = errors
                    turn_data["grid_after"] = deepcopy(self.grid)
                    self.consecutive_errors += 1
                    print(f"    {pfx} -> R{mv['row']+1}C{mv['col']+1} = {mv['value']} [INVALID: {errors[0][:60]}]", flush=True)
            else:
                turn_data["validation_errors"] = ["Could not parse MOVE from response"]
                turn_data["grid_after"] = deepcopy(self.grid)
                self.consecutive_errors += 1
                self.parse_failures += 1
                print(f"    {pfx} no MOVE parsed (consecutive errors: {self.consecutive_errors})", flush=True)

            self.turns.append(turn_data)

            if is_solved(self.grid):
                print(f"    {pfx} Grid solved!", flush=True)
                break
            if self.consecutive_errors >= config.MAX_CONSECUTIVE_ERRORS:
                print(f"    {pfx} Too many consecutive errors ({self.consecutive_errors}) — stopping", flush=True)
                break
            if self.parse_failures >= config.MAX_PARSE_RETRIES:
                print(f"    {pfx} Too many parse failures ({self.parse_failures}) — stopping", flush=True)
                break

    def _run_single_shot(self, log_prefix, start_time):
        pfx = log_prefix or f"[{self.model} single-shot r{self.run_number}]"
        print(f"  {pfx} Requesting full solution...", flush=True)
        prompt = build_single_shot_prompt(self.puzzle)

        grid_before = deepcopy(self.grid)
        api_response = self._call_api(prompt, pfx)

        turn_data = {
            "turn": 1,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "raw_response": api_response,
            "reasoning_raw": "",
            "move_raw": "",
            "parsed_move": None,
            "stuck": False,
            "valid": False,
            "validation_errors": [],
            "grid_before": grid_before,
            "grid_after": grid_before,
        }

        if api_response is None:
            turn_data["validation_errors"] = ["API call failed"]
            self.turns.append(turn_data)
            print(f"    {pfx} API failed, giving up", flush=True)
            return

        full_grid = parse_full_grid(api_response["content"], self.size)
        turn_data["reasoning_raw"] = api_response["content"][:2000]

        if full_grid is None:
            turn_data["validation_errors"] = ["Could not parse a full grid from response"]
            self.turns.append(turn_data)
            print(f"    {pfx} could not parse full grid", flush=True)
            return

        full_grid = _normalize_grid(full_grid, self.size)
        clue_violations = [
            f"R{r+1}C{c+1} clue {self.clues[r][c]} overwritten with {full_grid[r][c]}"
            for r in range(self.size) for c in range(self.size)
            if self.clues[r][c] != 0 and full_grid[r][c] != self.clues[r][c]
        ]
        correct = not clue_violations and grids_equal(full_grid, self.solution)

        self.grid = full_grid
        turn_data["grid_after"] = deepcopy(full_grid)
        turn_data["valid"] = correct
        turn_data["validation_errors"] = clue_violations if not correct else []
        self.turns.append(turn_data)
        print(f"    {pfx} full grid parsed, correct={correct}", flush=True)

    def _build_result(self, start_time):
        solved = is_solved(self.grid)
        correct = solved and grids_equal(self.grid, self.solution)
        if self.protocol == "single-shot":
            solved = correct = bool(self.turns) and self.turns[-1]["valid"]

        return {
            "meta": {
                "run_id": self._run_id(),
                "puzzle_id": self.puzzle["puzzle_id"],
                "protocol": self.protocol,
                "strategy_id": self.strategy_id or self.protocol,
                "strategy_label": STRATEGIES.get(self.strategy_id, {}).get(
                    "label", "Single-Shot"
                ),
                "model": self.model,
                "run_number": self.run_number,
                "timestamp_start": start_time.isoformat(),
                "timestamp_end": datetime.now().isoformat(),
                "duration_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                "temperature": config.TEMPERATURE,
                "max_tokens": config.MAX_TOKENS,
            },
            "puzzle": {
                "size": self.size,
                "box_width": self.box_width,
                "box_height": self.box_height,
                "clues": self.clues,
                "solution": self.solution,
                "clue_count": sum(1 for r in self.clues for c in r if c != 0),
            },
            "result": {
                "solved": solved,
                "total_turns": len(self.turns),
                "total_errors": sum(1 for t in self.turns if not t["valid"]),
                "total_prompt_tokens": sum(
                    t.get("raw_response", {}).get("prompt_tokens", 0) for t in self.turns
                ),
                "total_completion_tokens": sum(
                    t.get("raw_response", {}).get("completion_tokens", 0) for t in self.turns
                ),
                "final_grid": self.grid,
                "correct_against_solution": correct,
            },
            "turns": self.turns,
            "errors": [t for t in self.turns if not t["valid"]],
        }

    def _call_api(self, prompt, pfx=""):
        last_error = None
        for attempt in range(1, config.MAX_API_RETRIES + 1):
            try:
                print(f"    {pfx} API call ({self.model})...", flush=True)
                t0 = time.time()
                resp = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS,
                )
                elapsed = time.time() - t0
                choice = resp.choices[0]
                msg = choice.message
                usage = resp.usage
                tok = usage.total_tokens if usage else 0
                print(f"    {pfx} API done ({elapsed:.1f}s, {tok} tokens, finish={choice.finish_reason})", flush=True)
                return {
                    "content": msg.content or "",
                    "reasoning_content": getattr(msg, "reasoning_content", None),
                    "finish_reason": choice.finish_reason,
                    "model": resp.model,
                    "id": resp.id,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                }
            except NON_RETRYABLE_ERRORS as e:
                print(f"  {pfx} non-retryable API error ({type(e).__name__}), giving up immediately: {e}")
                return None
            except Exception as e:
                last_error = str(e)
                if attempt < config.MAX_API_RETRIES:
                    wait = 2 ** attempt
                    print(f"  API error (attempt {attempt}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        print(f"  API failed after {config.MAX_API_RETRIES} attempts: {last_error}")
        return None


def _normalize_grid(grid, size):
    out = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            val = grid[r][c]
            out[r][c] = val if val is not None and val != 0 else 0
    return out
