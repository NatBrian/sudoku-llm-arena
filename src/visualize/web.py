"""Generate interactive web visualization page."""
import json
import os
from .. import config


def generate_web_page(output_path=None):
    if output_path is None:
        output_path = config.OUTPUT_DIR / "web" / "index.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    runs = load_all_runs()
    if not runs:
        print("No parsed data found. Run the swarm first.")
        return

    runs_json = json.dumps(runs)
    puzzles_json = json.dumps(list_puzzles(runs))
    strategies_json = json.dumps(list_strategies(runs))
    models_json = json.dumps(list_models(runs))

    html = HTML_TEMPLATE.format(
        runs_data=runs_json,
        puzzles_list=puzzles_json,
        strategies_list=strategies_json,
        models_list=models_json,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Web page saved: {output_path} ({size_kb:.0f} KB)")
    return output_path


def load_all_runs():
    runs = []
    parsed_dir = config.PARSED_DIR
    if not parsed_dir.exists():
        return runs
    for path in sorted(parsed_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
            runs.append(data)
    return runs


def list_puzzles(runs):
    seen = set()
    result = []
    for r in runs:
        pid = r.get("puzzle_id", "unknown")
        if pid not in seen:
            seen.add(pid)
            result.append(pid)
    return result


def list_strategies(runs):
    seen = set()
    result = []
    for r in runs:
        sid = r.get("strategy_id", "unknown")
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def list_models(runs):
    seen = set()
    result = []
    for r in runs:
        m = r.get("model", "unknown")
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sudoku Arena — LLM Solving Visualization</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
  .header {{ background: #1a1a2e; color: white; padding: 20px 40px; }}
  .header h1 {{ font-size: 24px; margin-bottom: 4px; }}
  .header p {{ font-size: 14px; opacity: 0.8; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .controls {{ grid-column: 1 / -1; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls select, .controls button {{ padding: 8px 16px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; background: white; cursor: pointer; }}
  .controls button {{ background: #1a1a2e; color: white; border: none; }}
  .controls button:hover {{ background: #16213e; }}
  .controls button:disabled {{ opacity: 0.5; cursor: default; }}
  .controls label {{ font-size: 13px; font-weight: 600; }}
  .grid-panel {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .grid-panel h2 {{ font-size: 16px; margin-bottom: 12px; }}
  .info-panel {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .info-panel h2 {{ font-size: 16px; margin-bottom: 12px; }}
  canvas {{ display: block; margin: 0 auto; max-width: 100%; }}
  .controls-row {{ display: flex; gap: 8px; align-items: center; margin-top: 12px; flex-wrap: wrap; }}
  input[type="range"] {{ flex: 1; min-width: 100px; }}
  .step-info {{ font-size: 13px; color: #666; }}
  .reasoning {{ margin-top: 12px; padding: 12px; background: #f8f9fa; border-radius: 8px; font-size: 14px; line-height: 1.5; min-height: 60px; }}
  .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }}
  .stat {{ padding: 8px 12px; background: #f8f9fa; border-radius: 6px; }}
  .stat-label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
  .stat-value {{ font-size: 18px; font-weight: 700; }}
  .summary {{ grid-column: 1 / -1; background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .summary h2 {{ font-size: 16px; margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f8f9fa; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
  .badge-ok {{ background: #c8e6c9; color: #2e7d32; }}
  .badge-fail {{ background: #ffcdd2; color: #c62828; }}
  @media (max-width: 800px) {{ .container {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="header">
  <h1>Sudoku Arena</h1>
  <p>LLM Sudoku Solving Tournament — Interactive Visualization</p>
</div>

<div class="container">
  <div class="controls">
    <label>Puzzle:</label>
    <select id="selPuzzle">{puzzles_list}</select>
    <label>Model:</label>
    <select id="selModel">{models_list}</select>
    <label>Strategy:</label>
    <select id="selStrategy">{strategies_list}</select>
    <label>Run:</label>
    <select id="selRun"><option>1</option></select>
    <button id="btnPrev">&#9664; Prev</button>
    <button id="btnNext">Next &#9654;</button>
    <button id="btnPlay">&#9654; Play</button>
  </div>

  <div class="grid-panel" id="gridPanel">
    <h2 id="gridTitle">Grid</h2>
    <canvas id="gridCanvas" width="400" height="400"></canvas>
    <div class="controls-row">
      <span class="step-info" id="stepInfo">Step 0 / 0</span>
      <input type="range" id="timeline" min="0" max="0" value="0">
    </div>
  </div>

  <div class="info-panel">
    <h2>Step Details</h2>
    <div class="reasoning" id="reasoning">Select a run to view reasoning.</div>
    <div class="stats">
      <div class="stat"><div class="stat-label">Status</div><div class="stat-value" id="statStatus">-</div></div>
      <div class="stat"><div class="stat-label">Steps</div><div class="stat-value" id="statSteps">-</div></div>
      <div class="stat"><div class="stat-label">Errors</div><div class="stat-value" id="statErrors">-</div></div>
      <div class="stat"><div class="stat-label">Tokens</div><div class="stat-value" id="statTokens">-</div></div>
    </div>
  </div>

  <div class="summary">
    <h2>All Runs Summary</h2>
    <div id="summaryTable"></div>
  </div>
</div>

<script>
const RUNS = {runs_data};
const PUZZLES = {puzzles_list};
const STRATEGIES = {strategies_list};
const MODELS = {models_list};

let currentRun = null;
let currentSteps = [];
let currentStep = 0;
let playInterval = null;

const selPuzzle = document.getElementById('selPuzzle');
const selModel = document.getElementById('selModel');
const selStrategy = document.getElementById('selStrategy');
const selRun = document.getElementById('selRun');
const btnPrev = document.getElementById('btnPrev');
const btnNext = document.getElementById('btnNext');
const btnPlay = document.getElementById('btnPlay');
const timeline = document.getElementById('timeline');
const canvas = document.getElementById('gridCanvas');
const ctx = canvas.getContext('2d');

PUZZLES.forEach(p => {{ selPuzzle.add(new Option(p, p)); }});
MODELS.forEach(m => {{ selModel.add(new Option(m, m)); }});
STRATEGIES.forEach(s => {{ selStrategy.add(new Option(s, s)); }});

function getFilteredRuns() {{
  const pid = selPuzzle.value;
  const mid = selModel.value;
  const sid = selStrategy.value;
  return RUNS.filter(r => r.puzzle_id === pid && r.model === mid && r.strategy_id === sid);
}}

function populateRuns() {{
  const runs = getFilteredRuns();
  selRun.innerHTML = '';
  if (runs.length === 0) {{ selRun.add(new Option('No data', '')); return; }}
  runs.forEach((r, i) => {{ selRun.add(new Option('Run ' + (i + 1), i)); }});
  selRun.value = 0;
  loadRun(runs[0]);
}}

function loadRun(run) {{
  if (!run) return;
  currentRun = run;
  currentSteps = run.steps || [];
  currentStep = 0;
  timeline.max = Math.max(0, currentSteps.length - 1);
  timeline.value = 0;
  updateStats();
  renderStep(0);
}}

function renderStep(idx) {{
  if (!currentRun || !currentSteps.length) return;
  const steps = currentSteps;
  if (idx < 0) idx = 0;
  if (idx >= steps.length) idx = steps.length - 1;
  currentStep = idx;

  const step = steps[idx];
  const grid = step.grid_after || step.grid_before;
  if (!grid) return;
  const size = grid.length;
  const move = step.parsed_move;
  const backtrack = step.backtrack;
  const valid = step.valid;
  const clueGrid = steps.length > 0 && steps[0].grid_before ? steps[0].grid_before : grid;

  const total = steps.length;
  document.getElementById('stepInfo').textContent = 'Step ' + (idx + 1) + ' / ' + total;
  timeline.value = idx;

  const cellSize = canvas.width / size;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const boxW = size === 4 ? 2 : (size === 6 ? 3 : 3);
  const boxH = size === 4 ? 2 : (size === 6 ? 2 : 3);

  for (let r = 0; r < size; r++) {{
    for (let c = 0; c < size; c++) {{
      const val = grid[r][c];
      const isClue = (idx === 0) ? (val !== 0) : (clueGrid[r][c] !== 0);
      const isError = move && !valid && move.row === r && move.col === c;
      const isHighlight = (move && valid && move.row === r && move.col === c) ||
        (backtrack && valid && !move && backtrack.row === r && backtrack.col === c);
      const frac = total > 1 ? idx / (total - 1) : 0;

      let color = '#ffffff';
      if (isError) color = '#ff5252';
      else if (isClue) color = '#f0f0f0';
      else if (val !== 0) {{
        const R = Math.round(227 + (255 - 227) * frac);
        const G = Math.round(242 - (242 - 224) * frac);
        const B = Math.round(253 - (253 - 157) * frac);
        color = 'rgb(' + R + ',' + G + ',' + B + ')';
      }}

      ctx.fillStyle = color;
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);

      if (isHighlight) {{
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 3;
        ctx.strokeRect(c * cellSize + 1, r * cellSize + 1, cellSize - 2, cellSize - 2);
      }}

      ctx.strokeStyle = '#ccc';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);

      if (val !== 0) {{
        ctx.fillStyle = isClue ? '#333' : '#444';
        ctx.font = 'bold ' + (cellSize * 0.5) + 'px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val, c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
      }}
    }}
  }}

  for (let r = 0; r <= size; r++) {{
    ctx.strokeStyle = (r % boxH === 0) ? '#111' : '#ccc';
    ctx.lineWidth = (r % boxH === 0) ? 2.5 : 0.5;
    ctx.beginPath();
    ctx.moveTo(0, r * cellSize);
    ctx.lineTo(canvas.width, r * cellSize);
    ctx.stroke();
  }}
  for (let c = 0; c <= size; c++) {{
    ctx.strokeStyle = (c % boxW === 0) ? '#111' : '#ccc';
    ctx.lineWidth = (c % boxW === 0) ? 2.5 : 0.5;
    ctx.beginPath();
    ctx.moveTo(c * cellSize, 0);
    ctx.lineTo(c * cellSize, canvas.height);
    ctx.stroke();
  }}

  document.getElementById('gridTitle').textContent =
    (currentRun.strategy_label || currentRun.strategy_id) + ' [' + currentRun.model + '] — ' + currentRun.puzzle_id;

  const reasoning = step.reasoning || '(no reasoning)';
  document.getElementById('reasoning').textContent = reasoning;
}}

function updateStats() {{
  if (!currentRun) return;
  const r = currentRun;
  document.getElementById('statStatus').innerHTML =
    r.solved ? '<span class="badge badge-ok">Solved</span>' : '<span class="badge badge-fail">Unsolved</span>';
  document.getElementById('statSteps').textContent = r.total_turns || '-';
  document.getElementById('statErrors').textContent = r.total_errors || '-';
  document.getElementById('statTokens').textContent = (r.total_prompt_tokens + r.total_completion_tokens) || '-';
}}

function renderSummary() {{
  const table = document.getElementById('summaryTable');
  if (RUNS.length === 0) {{ table.innerHTML = '<p>No data yet.</p>'; return; }}
  let html = '<table><thead><tr><th>Puzzle</th><th>Model</th><th>Strategy</th><th>Run</th><th>Result</th><th>Turns</th><th>Errors</th><th>Tokens</th></tr></thead><tbody>';
  RUNS.forEach(r => {{
    const ok = r.solved ? 'badge badge-ok' : 'badge badge-fail';
    const lbl = r.solved ? 'Solved' : 'Failed';
    html += '<tr><td>' + (r.puzzle_id || '-') + '</td><td>' + (r.model || '-') + '</td><td>' + (r.strategy_label || r.strategy_id) + '</td><td>' + (r.run_number || '-') + '</td><td><span class="' + ok + '">' + lbl + '</span></td><td>' + (r.total_turns || '-') + '</td><td>' + (r.total_errors || '-') + '</td><td>' + (r.total_prompt_tokens + r.total_completion_tokens || '-') + '</td></tr>';
  }});
  html += '</tbody></table>';
  table.innerHTML = html;
}}

selPuzzle.addEventListener('change', populateRuns);
selModel.addEventListener('change', populateRuns);
selStrategy.addEventListener('change', populateRuns);
selRun.addEventListener('change', () => {{
  const runs = getFilteredRuns();
  const idx = parseInt(selRun.value);
  if (idx >= 0 && idx < runs.length) loadRun(runs[idx]);
}});

btnPrev.addEventListener('click', () => {{
  if (currentSteps.length) renderStep(currentStep - 1);
}});
btnNext.addEventListener('click', () => {{
  if (currentSteps.length) renderStep(currentStep + 1);
}});
btnPlay.addEventListener('click', () => {{
  if (playInterval) {{
    clearInterval(playInterval);
    playInterval = null;
    btnPlay.innerHTML = '&#9654; Play';
    return;
  }}
  btnPlay.innerHTML = '&#9646;&#9646; Pause';
  playInterval = setInterval(() => {{
    const next = currentStep + 1;
    if (next >= currentSteps.length) {{
      clearInterval(playInterval);
      playInterval = null;
      btnPlay.innerHTML = '&#9654; Play';
      return;
    }}
    renderStep(next);
  }}, 1200);
}});
timeline.addEventListener('input', () => {{
  renderStep(parseInt(timeline.value));
}});

populateRuns();
renderSummary();
</script>
</body>
</html>"""
