"""Download a W&B/Forge run and generate PNG and SVG graphs.

Example:
    python export_run.py "entity/project/run_id"
    python export_run.py "entity/project/run_id" --x-key env_steps

"""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib
matplotlib.use("Agg")  # Save images without requiring a GUI.
import matplotlib.pyplot as plt


def is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def download_run(run, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / 'config.json', dict(run.config))
    save_json(run_dir / 'summary.json', dict(run.summary))
    save_json(run_dir / 'metadata.json', {
        'id': run.id,
        'name': run.name,
        'state': run.state,
        'url': run.url,
        'entity': run.entity,
        'project': run.project,
    })

    print('Exporting metrics history...')
    history_path = run_dir / 'history.jsonl'
    count = 0
    # Publish the history only after all pages have been downloaded.
    history_file = NamedTemporaryFile(mode='w', encoding='utf-8', dir=run_dir,
                                      suffix='.jsonl.tmp', delete=False)
    temp_path = Path(history_file.name)
    try:
        with history_file:
            for row in run.scan_history(page_size=1000):
                history_file.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                count += 1
        temp_path.replace(history_path)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f'Exported {count} history rows')

    files_dir = run_dir / 'files'
    files_dir.mkdir(exist_ok=True)
    max_size = 100 * 1024 * 1024  # 100 MiB
    skipped = []
    for file in run.files():
        if file.size and file.size > max_size:
            skipped.append(file.name)
            continue
        try:
            print(f'Downloading: {file.name}')
            file.download(root=str(files_dir), replace=True)
        except Exception as exc:
            skipped.append(f'{file.name}: {exc}')
    (run_dir / 'skipped_files.txt').write_text('\n'.join(skipped), encoding='utf-8')
    print(f'Exported run data to {run_dir.resolve()}')
    return history_path


def plot_sample(points, max_points):
    """Reduce visual density while keeping high and low spikes in each bucket."""
    if len(points) <= max_points:
        return points

    bucket_size = math.ceil((len(points) - 2) / max(1, (max_points - 2) // 2))
    result = [points[0]]
    for start in range(1, len(points) - 1, bucket_size):
        chunk = points[start : min(start + bucket_size, len(points) - 1)]
        low = min(range(len(chunk)), key=lambda i: chunk[i][1])
        high = max(range(len(chunk)), key=lambda i: chunk[i][1])
        for i in sorted({low, high}):
            result.append(chunk[i])
    result.append(points[-1])
    return result


def safe_stem(name, used):
    stem = re.sub(r'[^a-zA-Z0-9._-]+', '_', name).strip('._')[:100] or 'metric'
    if stem.upper() in {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'LPT1'}:
        stem = f'metric_{stem}'
    if stem.lower() in used:
        suffix = hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]
        stem = f'{stem}_{suffix}'
    used.add(stem.lower())
    return stem


def export_graphs(history_path, graphs_dir, x_key, max_plot_points):
    graphs_dir.mkdir(parents=True, exist_ok=True)
    series = defaultdict(list)
    missing_x_rows = 0

    # Metrics may be logged at different frequencies. Only collect the
    # values actually present in each row, without filling missing values.
    with history_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            x = row.get(x_key)
            if not is_number(x):
                missing_x_rows += 1
                continue
            for key, value in row.items():
                if key == x_key or key.startswith('_'):
                    continue
                if is_number(value):
                    series[key].append((float(x), float(value)))

    print(f'Found {len(series)} numeric metrics with X = {x_key!r}')
    if missing_x_rows:
        print(f'Note: skipped {missing_x_rows} history rows without numeric {x_key!r}')

    index = {}
    used_stems = set()
    for key, points in sorted(series.items()):
        points.sort(key=lambda pair: pair[0])
        plotted = plot_sample(points, max_plot_points)
        stem = safe_stem(key, used_stems)
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        try:
            xs, ys = zip(*plotted)
            ax.plot(xs, ys, linewidth=1, marker='.' if len(plotted) == 1 else None)
            ax.set(title=key, xlabel=x_key, ylabel=key)
            ax.grid(alpha=0.25)
            for extension in ('png', 'svg'):
                fig.savefig(graphs_dir / f'{stem}.{extension}', dpi=150)
        finally:
            plt.close(fig)
        index[key] = {'png': f'{stem}.png', 'svg': f'{stem}.svg', 'points': len(points)}
    save_json(graphs_dir / 'index.json', index)
    return len(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run', metavar='entity/project/run_id')
    parser.add_argument('--output-dir', type=Path, default=Path('wandb_exports'),
                        help='Directory for downloaded runs and graphs')
    parser.add_argument('--x-key', default='_step')
    parser.add_argument('--max-plot-points', type=int, default=2000)
    args = parser.parse_args()

    parts = args.run.split('/')
    if len(parts) != 3 or not all(
        part not in {'.', '..'} and re.fullmatch(r'[\w.-]+', part) for part in parts
    ):
        parser.error('run must have the form entity/project/run_id')
    if args.max_plot_points < 4:
        parser.error('--max-plot-points must be at least 4')

    _, project, run_id = parts
    run_dir = args.output_dir / project / run_id
    import wandb

    run = wandb.Api().run(args.run)
    history_path = download_run(run, run_dir)

    graphs_dir = run_dir / 'graphs'
    count = export_graphs(history_path, graphs_dir, args.x_key, args.max_plot_points)
    if not count:
        parser.exit(1, f'No graphs generated. Check --x-key (currently {args.x_key!r}).\n')
    print(f'Saved {count} graphs in PNG and SVG to {graphs_dir.resolve()}')


if __name__ == '__main__':
    main()
