# Frozen V13 candidate grid

55 new workflows = 38 base + 17 fusion. Base candidate settings = 84.

| Workflow | Site | Representation | Learner | Candidates |
|---|---|---|---|---:|
| hgb_pc_oral | oral | pc | hgb | 2 |
| hgb_pc_balanced | balanced | pc | hgb | 2 |
| cat_pc_oral | oral | pc | cat | 2 |
| cat_pc_balanced | balanced | pc | cat | 2 |
| mars_pc_oral | oral | pc | mars | 2 |
| mars_pc_balanced | balanced | pc | mars | 2 |
| gp_pc_oral | oral | pc | gp | 3 |
| gp_pc_balanced | balanced | pc | gp | 3 |
| hgb_sir_oral | oral | sir | hgb | 2 |
| hgb_sir_balanced | balanced | sir | hgb | 2 |
| cat_sir_oral | oral | sir | cat | 2 |
| cat_sir_balanced | balanced | sir | cat | 2 |
| mars_sir_oral | oral | sir | mars | 2 |
| mars_sir_balanced | balanced | sir | mars | 2 |
| gp_sir_oral | oral | sir | gp | 3 |
| gp_sir_balanced | balanced | sir | gp | 3 |
| hgb_save_oral | oral | save | hgb | 2 |
| hgb_save_balanced | balanced | save | hgb | 2 |
| cat_save_oral | oral | save | cat | 2 |
| cat_save_balanced | balanced | save | cat | 2 |
| mars_save_oral | oral | save | mars | 2 |
| mars_save_balanced | balanced | save | mars | 2 |
| gp_save_oral | oral | save | gp | 3 |
| gp_save_balanced | balanced | save | gp | 3 |
| bin_cat_pc_oral | oral | pc | bin_cat | 2 |
| bin_cat_pc_balanced | balanced | pc | bin_cat | 2 |
| bin_hgb_pc_oral | oral | pc | bin_hgb | 2 |
| bin_hgb_pc_balanced | balanced | pc | bin_hgb | 2 |
| quantile_cat_pc_oral | oral | pc | quantile_cat | 2 |
| quantile_cat_pc_balanced | balanced | pc | quantile_cat | 2 |
| bin_cat_save_oral | oral | save | bin_cat | 2 |
| bin_cat_save_balanced | balanced | save | bin_cat | 2 |
| bin_hgb_save_oral | oral | save | bin_hgb | 2 |
| bin_hgb_save_balanced | balanced | save | bin_hgb | 2 |
| quantile_cat_save_oral | oral | save | quantile_cat | 2 |
| quantile_cat_save_balanced | balanced | save | quantile_cat | 2 |
| graph_oral | oral | graph | graph | 3 |
| graph_balanced | balanced | graph | graph | 3 |

## Shared learner grids

Each JSON object is one candidate. Config is authoritative.

### hgb

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "iterations": 180,
    "leaves": 7,
    "leaf": 40,
    "l2": 30
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "iterations": 280,
    "leaves": 15,
    "leaf": 40,
    "l2": 30
  }
]
```

### cat

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "iterations": 600,
    "depth": 4,
    "l2": 30
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "iterations": 800,
    "depth": 6,
    "l2": 30
  }
]
```

### mars

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "max_terms": 13,
    "degree": 1,
    "penalty": 0.01
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "max_terms": 25,
    "degree": 2,
    "penalty": 0.05
  }
]
```

### gp

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "inducing": 96,
    "length": 1.0,
    "noise": 0.1
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "inducing": 192,
    "length": 0.7,
    "noise": 0.05
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "inducing": 192,
    "length": 1.5,
    "noise": 0.2
  }
]
```

### bin_cat

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "iterations": 600,
    "depth": 4,
    "l2": 30,
    "bins": 4
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "iterations": 800,
    "depth": 6,
    "l2": 30,
    "bins": 8
  }
]
```

### bin_hgb

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "iterations": 180,
    "leaves": 7,
    "leaf": 40,
    "l2": 30,
    "bins": 4
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "iterations": 280,
    "leaves": 15,
    "leaf": 40,
    "l2": 30,
    "bins": 8
  }
]
```

### quantile_cat

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "iterations": 600,
    "depth": 4,
    "l2": 30,
    "quantiles": 5
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "iterations": 800,
    "depth": 6,
    "l2": 30,
    "quantiles": 9
  }
]
```

### graph

```json
[
  {
    "count": 128,
    "penalty": 0.01,
    "graph_weight": 0.0
  },
  {
    "count": 128,
    "penalty": 0.01,
    "graph_weight": 1.0
  },
  {
    "count": 128,
    "penalty": 0.01,
    "graph_weight": 10.0
  }
]
```

## Fusion pools

| Pool | Members |
|---|---:|
| oral | 19 |
| balanced | 19 |
| non_tree | 14 |
| new | 38 |
| combined | 128 |

Convex/ridge/positive on all five pools, sparse on new/combined only.

Ridge/positive penalties: [0.003, 0.01, 0.1, 1]

Sparse penalties: [0.001, 0.01, 0.1]

V11 replay is a reference, not a new workflow. All penalties use current outer-training data only.
