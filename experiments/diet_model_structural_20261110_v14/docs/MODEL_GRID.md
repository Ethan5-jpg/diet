# V14 frozen model grid

40 base workflows, 80 base candidate fits per role, 26 meta workflows; 66 workflows total.
Two settings per base family; nonlinear meta models have two internal settings per pool.
Historical reference/replay workflows are excluded from these counts.

## Base workflows

| Name | Learner | Representation | Site | Candidate settings |
|---|---|---|---|---:|
| hgb_sir_oral | hgb | sir | oral | 2 |
| hgb_sir_balanced | hgb | sir | balanced | 2 |
| cat_sir_oral | cat | sir | oral | 2 |
| cat_sir_balanced | cat | sir | balanced | 2 |
| hgb_rff_sir_oral | hgb | rff_sir | oral | 2 |
| hgb_rff_sir_balanced | hgb | rff_sir | balanced | 2 |
| cat_rff_sir_oral | cat | rff_sir | oral | 2 |
| cat_rff_sir_balanced | cat | rff_sir | balanced | 2 |
| hgb_pilot_gradient_oral | hgb | pilot_gradient | oral | 2 |
| hgb_pilot_gradient_balanced | hgb | pilot_gradient | balanced | 2 |
| cat_pilot_gradient_oral | cat | pilot_gradient | oral | 2 |
| cat_pilot_gradient_balanced | cat | pilot_gradient | balanced | 2 |
| local_sir_oral | local | sir | oral | 2 |
| local_sir_balanced | local | sir | balanced | 2 |
| forest_local_sir_oral | forest_local | sir | oral | 2 |
| forest_local_sir_balanced | forest_local | sir | balanced | 2 |
| ntk_sir_oral | ntk | sir | oral | 2 |
| ntk_sir_balanced | ntk | sir | balanced | 2 |
| diffusion_sir_oral | diffusion | sir | oral | 2 |
| diffusion_sir_balanced | diffusion | sir | balanced | 2 |
| svr_sir_oral | svr | sir | oral | 2 |
| svr_sir_balanced | svr | sir | balanced | 2 |
| rbf_ard_sir_oral | rbf_ard | sir | oral | 2 |
| rbf_ard_sir_balanced | rbf_ard | sir | balanced | 2 |
| local_rff_sir_oral | local | rff_sir | oral | 2 |
| local_rff_sir_balanced | local | rff_sir | balanced | 2 |
| forest_local_rff_sir_oral | forest_local | rff_sir | oral | 2 |
| forest_local_rff_sir_balanced | forest_local | rff_sir | balanced | 2 |
| ntk_rff_sir_oral | ntk | rff_sir | oral | 2 |
| ntk_rff_sir_balanced | ntk | rff_sir | balanced | 2 |
| diffusion_rff_sir_oral | diffusion | rff_sir | oral | 2 |
| diffusion_rff_sir_balanced | diffusion | rff_sir | balanced | 2 |
| svr_rff_sir_oral | svr | rff_sir | oral | 2 |
| svr_rff_sir_balanced | svr | rff_sir | balanced | 2 |
| rbf_ard_rff_sir_oral | rbf_ard | rff_sir | oral | 2 |
| rbf_ard_rff_sir_balanced | rbf_ard | rff_sir | balanced | 2 |
| hgb_paired_pc_balanced | hgb | paired_pc | balanced | 2 |
| cat_paired_pc_balanced | cat | paired_pc | balanced | 2 |
| hgb_cca_balanced | hgb | cca | balanced | 2 |
| cat_cca_balanced | cat | cca | balanced | 2 |

### hgb

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "pre_components": 24,
    "rff_features": 128,
    "bandwidth": 1.0,
    "pilot_penalty": 0.05,
    "cca_penalty": 0.1,
    "iterations": 180,
    "leaves": 7,
    "leaf": 40,
    "l2": 30
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "pre_components": 32,
    "rff_features": 192,
    "bandwidth": 0.7,
    "pilot_penalty": 0.1,
    "cca_penalty": 0.5,
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
    "pre_components": 24,
    "rff_features": 128,
    "bandwidth": 1.0,
    "pilot_penalty": 0.05,
    "cca_penalty": 0.1,
    "iterations": 600,
    "depth": 4,
    "l2": 30
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "pre_components": 32,
    "rff_features": 192,
    "bandwidth": 0.7,
    "pilot_penalty": 0.1,
    "cca_penalty": 0.5,
    "iterations": 800,
    "depth": 6,
    "l2": 30
  }
]
```

### local

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "pre_components": 24,
    "rff_features": 128,
    "bandwidth": 1.0,
    "pilot_penalty": 0.05,
    "cca_penalty": 0.1,
    "neighbors": 80,
    "penalty": 0.1
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "pre_components": 32,
    "rff_features": 192,
    "bandwidth": 0.7,
    "pilot_penalty": 0.1,
    "cca_penalty": 0.5,
    "neighbors": 160,
    "penalty": 0.5
  }
]
```

### forest_local

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "pre_components": 24,
    "rff_features": 128,
    "bandwidth": 1.0,
    "pilot_penalty": 0.05,
    "cca_penalty": 0.1,
    "trees": 64,
    "depth": 5,
    "leaf": 25,
    "penalty": 0.1
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "pre_components": 32,
    "rff_features": 192,
    "bandwidth": 0.7,
    "pilot_penalty": 0.1,
    "cca_penalty": 0.5,
    "trees": 96,
    "depth": 7,
    "leaf": 35,
    "penalty": 0.5
  }
]
```

### ntk

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "pre_components": 24,
    "rff_features": 128,
    "bandwidth": 1.0,
    "pilot_penalty": 0.05,
    "cca_penalty": 0.1,
    "inducing": 96,
    "layers": 2,
    "penalty": 0.01
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "pre_components": 32,
    "rff_features": 192,
    "bandwidth": 0.7,
    "pilot_penalty": 0.1,
    "cca_penalty": 0.5,
    "inducing": 192,
    "layers": 3,
    "penalty": 0.05
  }
]
```

### diffusion

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "pre_components": 24,
    "rff_features": 128,
    "bandwidth": 1.0,
    "pilot_penalty": 0.05,
    "cca_penalty": 0.1,
    "neighbors": 20,
    "smoothing": 0.5,
    "trend_penalty": 0.1
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "pre_components": 32,
    "rff_features": 192,
    "bandwidth": 0.7,
    "pilot_penalty": 0.1,
    "cca_penalty": 0.5,
    "neighbors": 40,
    "smoothing": 2.0,
    "trend_penalty": 1.0
  }
]
```

### svr

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "pre_components": 24,
    "rff_features": 128,
    "bandwidth": 1.0,
    "pilot_penalty": 0.05,
    "cca_penalty": 0.1,
    "C": 1.0,
    "epsilon": 0.15,
    "gamma_scale": 1.0
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "pre_components": 32,
    "rff_features": 192,
    "bandwidth": 0.7,
    "pilot_penalty": 0.1,
    "cca_penalty": 0.5,
    "C": 5.0,
    "epsilon": 0.25,
    "gamma_scale": 0.5
  }
]
```

### rbf_ard

```json
[
  {
    "count": 128,
    "components": 6,
    "slices": 8,
    "pre_components": 24,
    "rff_features": 128,
    "bandwidth": 1.0,
    "pilot_penalty": 0.05,
    "cca_penalty": 0.1,
    "inducing": 64,
    "length": 1.0,
    "max_iter": 300
  },
  {
    "count": 256,
    "components": 10,
    "slices": 12,
    "pre_components": 32,
    "rff_features": 192,
    "bandwidth": 0.7,
    "pilot_penalty": 0.1,
    "cca_penalty": 0.5,
    "inducing": 128,
    "length": 1.5,
    "max_iter": 500
  }
]
```

## Meta workflows

| Name | Pool size |
|---|---:|
| convex_oral | 18 |
| ridge_oral | 18 |
| positive_oral | 18 |
| convex_balanced | 22 |
| ridge_balanced | 22 |
| positive_balanced | 22 |
| convex_non_tree | 20 |
| ridge_non_tree | 20 |
| positive_non_tree | 20 |
| convex_new | 40 |
| ridge_new | 40 |
| positive_new | 40 |
| convex_bank | 128 |
| ridge_bank | 128 |
| positive_bank | 128 |
| convex_combined | 168 |
| ridge_combined | 168 |
| positive_combined | 168 |
| sparse_new | 40 |
| sparse_combined | 168 |
| local_stack_bank | 128 |
| ntk_stack_bank | 128 |
| rbf_ard_stack_bank | 128 |
| local_stack_combined | 168 |
| ntk_stack_combined | 168 |
| rbf_ard_stack_combined | 168 |

### Linear penalties

```json
{
  "meta_penalties": [
    0.003,
    0.01,
    0.1,
    1
  ],
  "sparse_penalties": [
    0.001,
    0.01,
    0.1
  ]
}
```

### Nonlinear meta settings

```json
{
  "local": [
    {
      "neighbors": 100,
      "penalty": 0.3,
      "meta_components": 12
    },
    {
      "neighbors": 180,
      "penalty": 1.0,
      "meta_components": 20
    }
  ],
  "ntk": [
    {
      "inducing": 96,
      "layers": 2,
      "penalty": 0.03,
      "meta_components": 16
    },
    {
      "inducing": 128,
      "layers": 3,
      "penalty": 0.1,
      "meta_components": 24
    }
  ],
  "rbf_ard": [
    {
      "inducing": 48,
      "length": 1.0,
      "max_iter": 300,
      "meta_components": 12
    },
    {
      "inducing": 96,
      "length": 1.5,
      "max_iter": 500,
      "meta_components": 20
    }
  ]
}
```

## Prespecified matched diagnostics

| Model | Comparator |
|---|---|
| hgb_rff_sir_oral | hgb_sir_oral |
| hgb_rff_sir_balanced | hgb_sir_balanced |
| cat_rff_sir_oral | cat_sir_oral |
| cat_rff_sir_balanced | cat_sir_balanced |
| hgb_pilot_gradient_oral | hgb_sir_oral |
| hgb_pilot_gradient_balanced | hgb_sir_balanced |
| cat_pilot_gradient_oral | cat_sir_oral |
| cat_pilot_gradient_balanced | cat_sir_balanced |
| local_rff_sir_oral | local_sir_oral |
| local_rff_sir_balanced | local_sir_balanced |
| forest_local_rff_sir_oral | forest_local_sir_oral |
| forest_local_rff_sir_balanced | forest_local_sir_balanced |
| ntk_rff_sir_oral | ntk_sir_oral |
| ntk_rff_sir_balanced | ntk_sir_balanced |
| diffusion_rff_sir_oral | diffusion_sir_oral |
| diffusion_rff_sir_balanced | diffusion_sir_balanced |
| svr_rff_sir_oral | svr_sir_oral |
| svr_rff_sir_balanced | svr_sir_balanced |
| rbf_ard_rff_sir_oral | rbf_ard_sir_oral |
| rbf_ard_rff_sir_balanced | rbf_ard_sir_balanced |
| hgb_cca_balanced | hgb_paired_pc_balanced |
| cat_cca_balanced | cat_paired_pc_balanced |
| convex_balanced | convex_oral |
| convex_combined | convex_new |
| ridge_balanced | ridge_oral |
| ridge_combined | ridge_new |
| positive_balanced | positive_oral |
| positive_combined | positive_new |
| local_stack_bank | positive_bank |
| ntk_stack_bank | positive_bank |
| rbf_ard_stack_bank | positive_bank |
| local_stack_combined | positive_combined |
| ntk_stack_combined | positive_combined |
| rbf_ard_stack_combined | positive_combined |

All models are tuned using current outer-training household folds. Native performance is pending.
