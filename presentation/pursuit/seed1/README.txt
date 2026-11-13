Adversarial Pursuit — static ExpoComm, training seed 1

Data
- data/seed1_evaluations.csv: all 303 evaluation points for 25/45/61 agents.
  Columns identify the team, map, seed, training timestep, mean return and episode SD.
- data/agents_{25,45,61}_seed1_eval.csv: the same trajectories in separate files.
- data/seed1_run_summary.csv: final results, gameplay-episode returns and run links.
- data/seed1_behaviour_metrics.csv: six initial/final probe rows, with returns,
  tag counts and percentages. The accompanying JSON retains the source values.

Figures and visualisations
- figures/test_return_mean_seed1.{png,pdf,svg}: three learning curves.
- figures/behaviour_seed1.{png,pdf,svg}: boundary tagging and probe statistics.
- gifs/agents_{25,45,61}_seed1_final.gif: final-policy gameplay, one episode each.
- storyboards/agents_{25,45,61}_seed1_storyboard.png: frames at steps 0/100/200.

Replot with your own theme
Install matplotlib, then run from this folder:
  python replot.py
  python replot.py --style dark_background --color "#FFD166" --output my_theme
A custom .mplstyle file also works. The script uses only the bundled CSVs and
exports return curves and behaviour-statistics charts as PNG/PDF/SVG. It leaves
the supplied figures unchanged; edit the script for other layouts or labels.

Notes
- Returns: 100 greedy episodes per point; SD is across episodes, not training seeds.
- Behaviour: one matched episode per size; rewarded tags are inferred events,
  not unique prey captures. GIF episode returns differ from benchmark means.
- test_red_team_win_mean is unavailable for these Pursuit runs.
- manifest.json lists files, hashes and provenance. All measured results use seed 1.
