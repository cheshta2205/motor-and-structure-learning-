# Manuscript figures: SRO, OLM/ORM and Rochefort

Code and data for the three hippocampal/cerebellar sections of the paper. Every
figure listed below regenerates from what is in this folder — no retraining
required.

Paths in the scripts are resolved relative to the script's own directory, so run
them from inside `published_simulations/`. Checkpoints land in `checkpoints/` and figures in
`figures/main/<TASK>/`.

```
pip install numpy torch matplotlib scikit-learn
cd published_simulations
```

## Figure map

Figure numbers follow the manuscript, **not** the filenames in the figure
archive (there, Figure 7 and Figure 8 are swapped).

| Paper figure | What it shows | Script |
|---|---|---|
| Figure 7a,b | OLM/ORM paradigm schematics | `olm_orm_task_schematic.py` |
| Figure 7c–f | Prediction error, changed vs unchanged object | `gap_rows_fig.py` |
| Figure 7g,h | Preference index vs perturbation strength | `olm_orm_final.py --figures` |
| Figure 8a | SRO task table | `sro_task_schematic.py` |
| Figure 8b | MTL state geometry in PCA space | `sro_geometry_pca.py` |
| Figure 8c,d | Context decoding and CCGP | `sro_paper_figs.py` |
| ED Figure 8 | Dissociation vs perturbation strength and window | `pi_tau_half.py` |
| ED Figure 9 | Robustness to the observation model | `olm_orm_robustness_sweep.py --figure` |
| ED Figure 10 | Place-field stability under cerebellar perturbation | `roche_pooled_final.py` |
| ED Figure 11 | Geometry across MTL and cerebellar layers | `sro_paper_figs.py` |

## Regenerating the figures

**SRO.** Every SRO script needs `SRO_VARIANT=bernardi`. Without it the scripts
silently run a different task table (`full`, where reinforcement reverses on all
four stimuli) and the numbers will not match the paper.

```bash
SRO_VARIANT=bernardi python sro_task_schematic.py                  # Fig 8a
SRO_VARIANT=bernardi python sro_geometry_pca.py --dims 3           # Fig 8b
SRO_VARIANT=bernardi python sro_paper_figs.py --seeds 0 1 2 3 4 5 6 7 8 9
                                                                   # Fig 8c,d + ED 11
```

**OLM/ORM.**

```bash
python olm_orm_task_schematic.py                                   # Fig 7a,b
python gap_rows_fig.py                                             # Fig 7c-f
python pi_tau_half.py                                              # ED 8
python olm_orm_robustness_sweep.py --figure                        # ED 9
```

**Rochefort.**

```bash
python roche_pooled_final.py --seeds 0 1 2 3 4 5 6 7 8 9           # ED 10
```

## What is included

`checkpoints/` (25 MB)

- `rochefort_final_wt_s{0..9}.pt` and `rochefort_final_l7pkci_s{0..9}_sig1.00.pt`
  — ED 10 re-simulates from the trained networks rather than reading a cache.
- `sro_match_bernardi/sro_main_seed0.pt` — the single network Figure 8b projects.

`figures/main/` (15 MB) — cached results the figure scripts read:

- `OLM_ORM_FINAL/pooled_ac_seeds/pooled.npz` — Figure 7c–f
- `OLM_ORM_FINAL/pooled_ac_seeds/pi_surface.npz` — ED 8
- `OLM_ORM_FINAL/robustness/sweep_s*.npz`, `decompose*.npz` — ED 9
- `SRO_match_bernardi/seed*/` — Figure 8c,d and ED 11

## Retraining from scratch

The checkpoints for retraining are not committed; these are the commands that
produce them. Each is hours of CPU time.

```bash
python olm_orm_final.py --train --seeds 0 1 2 3 4 5 6 7 8 9
python olm_orm_final.py --figures --seeds 0 1 2 3 4 5 6 7 8 9     # writes pooled.npz
python pi_surface.py --measure --seeds 0 1 2 3 4 5 6 7 8 9        # writes pi_surface.npz

python olm_orm_robustness_sweep.py --train --seed 0               # repeat for seeds 1, 2
python olm_orm_robustness_sweep.py --measure --seed 0
python olm_orm_robustness_sweep.py --decompose --seed 0

python rochefort_final.py --train --seeds 0 1 2 3 4 5 6 7 8 9

SRO_VARIANT=bernardi python sro.py --train --arms main control --seeds 0 1 2 3 4 5 6 7 8 9
SRO_VARIANT=bernardi python sro_behaviour_pooled.py --measure --seeds 0 1 2 3 4 5 6 7 8 9
SRO_VARIANT=bernardi python sro_geometry_main_pooled.py --measure --seeds 0 1 2 3 4 5 6 7 8 9
SRO_VARIANT=bernardi python sro_geometry_pooled.py
```

`sro_behaviour_pooled.py`, `sro_geometry_main_pooled.py` and
`sro_geometry_pooled.py` produce no figures of their own — they write the
per-seed `.npz`/`.json` files that `sro_paper_figs.py` reads.

## Notes

- `sro_geometry_pca.py` is the panel-b script; an older MDS version was used
  during development and is not included.
- `gap_rows_fig.py` and `pi_tau_half.py` do not simulate anything; they read the
  cached `.npz` files and are the fastest way to check the figures reproduce.
- Numbers quoted in the text come from these same runs: ten networks per
  condition, mean ± s.e.m. across networks.
