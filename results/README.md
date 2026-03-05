# Results

## Simulator Realism

![Simulator realism](simulator_realism.png)

## Belief State DB — Hierarchical Heatmap

![Belief state heatmap](state_belief_db_hierarchical_heatmap_avg.png)

---

## Paper Run

### Training Curves

![Training curves](paper_run/training_curves.png)

### Benchmark Rewards

![Benchmark rewards](paper_run/benchmark_rewards.png)

```
Benchmark — 100 evaluation seeds
--------------------------------------------------------
condition             mean      std   d_taylor
--------------------------------------------------------
taylor_rule        -647.66   790.60          —
baseline           -701.03   585.65     -53.36
llm                -519.19   608.75    +128.48
oracle             -592.81   647.30     +54.85
--------------------------------------------------------
```

### Benchmark Trajectories

| No Shock | Min Shock |
|---|---|
| ![No shock](paper_run/benchmark_trajectories_no_shock.png) | ![Min shock](paper_run/benchmark_trajectories_min_shock.png) |

| Standard Shock | Intense Shock |
|---|---|
| ![Standard shock](paper_run/benchmark_trajectories_standard_shock.png) | ![Intense shock](paper_run/benchmark_trajectories_intense_shock.png) |

### COVID Evaluation

![COVID trajectories](paper_run/covid_trajectories.png)

```
COVID Benchmark — 1 runs each
--------------------------------------------------------
condition             mean      std   d_taylor
--------------------------------------------------------
taylor_rule       -1678.37     0.00          —
baseline          -1610.02     0.00     +68.35
llm               -1469.07     0.00    +209.30
oracle            -2020.33     0.00    -341.96
--------------------------------------------------------
```
