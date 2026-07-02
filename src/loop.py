import time
from copy import deepcopy
from datetime import datetime
from openai import OpenAI

from . import config
from .validator import validate_move, is_solved, count_empty, grids_equal
from .parser import parse_response, is_stuck_response
from .strategies import build_prompt, STRATEGIES
from .storage import save_raw_run


class GameLoop:
    def __init__(self, puzzle, strategy_id, run_number):
        self.puzzle = puzzle
        self.strategy_id = strategy_id
        self.run_number = run_number
        if not config.API_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable is required."
            )
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.API_BASE_URL,
        )
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
        self.move_history = set()

    def run(self, log_prefix=""):
        start_time = datetime.now()
        pfx = log_prefix or f"[{self.strategy_id} r{self.run_number}]"

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
                "candidates_before": self._compute_candidates(),
            }

            if parsed["stuck"]:
                turn_data["validation_errors"] = ["LLM reported being stuck"]
                turn_data["grid_after"] = deepcopy(self.grid)
                self.turns.append(turn_data)
                print(f"    {pfx} LLM is stuck — ending run", flush=True)
                break

            if parsed["move"]:
                mv = parsed["move"]
                errors = validate_move(
                    self.grid,
                    mv["row"],
                    mv["col"],
                    mv["value"],
                    self.box_width,
                    self.box_height,
                )
                if not errors:
                    self.grid[mv["row"]][mv["col"]] = mv["value"]
                    turn_data["valid"] = True
                    turn_data["grid_after"] = deepcopy(self.grid)
                    turn_data["validation_errors"] = []
                    self.consecutive_errors = 0
                    self.move_history.add(
                        (mv["row"], mv["col"], mv["value"])
                    )
                    print(f"    {pfx} -> R{mv['row']+1}C{mv['col']+1} = {mv['value']} [valid]", flush=True)
                else:
                    turn_data["validation_errors"] = errors
                    turn_data["grid_after"] = deepcopy(self.grid)
                    self.consecutive_errors += 1
                    print(f"    {pfx} -> R{mv['row']+1}C{mv['col']+1} = {mv['value']} [INVALID: {errors[0][:60]}]", flush=True)
            else:
                turn_data["validation_errors"] = [
                    "Could not parse MOVE from response"
                ]
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

        solved = is_solved(self.grid)
        result = {
            "meta": {
                "run_id": f"{self.puzzle['puzzle_id']}_{self.strategy_id}_r{self.run_number}",
                "puzzle_id": self.puzzle["puzzle_id"],
                "strategy_id": self.strategy_id,
                "strategy_label": STRATEGIES[self.strategy_id]["label"],
                "run_number": self.run_number,
                "timestamp_start": start_time.isoformat(),
                "timestamp_end": datetime.now().isoformat(),
                "duration_ms": int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
                "model": config.MODEL,
                "temperature": config.TEMPERATURE,
                "max_tokens": config.MAX_TOKENS,
                "base_url": config.API_BASE_URL,
            },
            "puzzle": {
                "size": self.size,
                "box_width": self.box_width,
                "box_height": self.box_height,
                "clues": self.clues,
                "solution": self.solution,
                "clue_count": sum(
                    1 for r in self.clues for c in r if c != 0
                ),
            },
            "result": {
                "solved": solved,
                "total_turns": len(self.turns),
                "total_errors": sum(
                    1 for t in self.turns if not t["valid"]
                ),
                "total_prompt_tokens": sum(
                    t.get("raw_response", {}).get("prompt_tokens", 0)
                    for t in self.turns
                ),
                "total_completion_tokens": sum(
                    t.get("raw_response", {}).get("completion_tokens", 0)
                    for t in self.turns
                ),
                "final_grid": self.grid,
                "correct_against_solution": solved
                and grids_equal(self.grid, self.solution),
            },
            "turns": self.turns,
            "errors": [t for t in self.turns if not t["valid"]],
        }

        save_raw_run(result)
        return result

    def _call_api(self, prompt, pfx=""):
        last_error = None
        for attempt in range(1, config.MAX_API_RETRIES + 1):
            try:
                print(f"    {pfx} API call...", flush=True)
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=config.MODEL,
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
                    "reasoning_content": getattr(
                        msg, "reasoning_content", None
                    ),
                    "finish_reason": choice.finish_reason,
                    "model": resp.model,
                    "id": resp.id,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens
                    if usage
                    else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                }
            except Exception as e:
                last_error = str(e)
                if attempt < config.MAX_API_RETRIES:
                    wait = 2 ** attempt
                    print(f"  API error (attempt {attempt}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        print(f"  API failed after {config.MAX_API_RETRIES} attempts: {last_error}")
        return None

    def _compute_candidates(self):
        candidates = {}
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == 0:
                    possible = []
                    for v in range(1, self.size + 1):
                        errs = validate_move(
                            self.grid,
                            r,
                            c,
                            v,
                            self.box_width,
                            self.box_height,
                        )
                        if not errs:
                            possible.append(v)
                    candidates[f"R{r+1}C{c+1}"] = possible
        return candidates


def _normalize_grid(grid, size):
    out = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            val = grid[r][c]
            out[r][c] = val if val is not None and val != 0 else 0
    return out
