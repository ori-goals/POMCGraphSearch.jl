# Partially Observable Monte-Carlo Graph Search

## Overview

This repository contains the **Julia implementation** of the ICAPS 2025 paper:  
> **Partially Observable Monte-Carlo Graph Search**  
> *Yang You, Vincent Thomas, Alex Schutz, Robert Skilton, Nick Hawes, Olivier Buffet*

The repository implements POMCGS, an offline planning algorithm that performs Monte-Carlo Graph Search for POMDPs and stores policies as finite-state controllers (FSCs).

---

## Code Structure

| File | Description |
|------|-------------|
| `./POMCGS.jl` | Main POMCGS algorithm |
| `./DiscretePlanner.jl` | Discrete POMDP planner version |
| `./ContinuousPlanner.jl` | Continuous POMDP planner version |
| `./FSC.jl` | Defines the FSC data structure used during planning |
| `./Qlearning.jl` | Implements the MDP heuristic method |
| `./Utils.jl` | Utility functions |

---

## Getting Started

You will need [Julia](https://julialang.org/) installed to run the examples.

Then you need install POMCGS via
```Julia
using Pkg
Pkg.add("POMCGS")
```

### Example 1 — RockSample

```julia
using POMCGS

using RockSample  # include the RockSample problem

pomdp = RockSamplePOMDP(7, 8)  # define a RockSample problem

pomcgs = POMCGS(pomdp;
    max_b_gap = 0.1, # the belief merging threshold
    max_search_depth = 30,
    nb_process_action_samples = 1000 # the number of simulations needed for processing each action
)  # Initialize the POMCGS solver. It will automatically use the discrete planner for this problem.

Solve(pomcgs)  # solve the problem
```

---

### Example 2 — LightDark

```julia
using POMCGS 

using POMDPModels

pomdp = LightDark1D()  # define the LightDark problem

pomcgs = POMCGS(pomdp;
    max_b_gap = 0.2, 
    state_grid = [1.0, 1.0], # the state grid for state discretization
    num_fixed_observations = 10, # the number of observation clusters
    max_search_depth = 30,
    nb_process_action_samples = 5000
)  # Initialize the POMCGS solver. It will automatically use the continuous planner for this problem.

Solve(pomcgs)  # solve the problem
```

### Saving FSC Policies

```Julia
SaveFSCPolicyJSON(pomcgs.fsc) # save the fsc policy to a JSON file
```

or 

```Julia
SaveFSCPolicyJLD2(pomcgs.fsc) # save the fsc policy to a JLD2 file
```

---

## Citation

If you found this work useful in your research, please cite:

```
@article{You_Thomas_Schutz_Skilton_Hawes_Buffet_2025,
  title={Partially Observable Monte-Carlo Graph Search},
  author={You, Yang and Thomas, Vincent and Schutz, Alex and Skilton, Robert and Hawes, Nick and Buffet, Olivier},
  journal={Proceedings of the International Conference on Automated Planning and Scheduling},
  volume={35},
  number={1},
  pages={279--287},
  year={2025},
  month={Sep.},
  url={https://ojs.aaai.org/index.php/ICAPS/article/view/36129},
  doi={10.1609/icaps.v35i1.36129}
}
```
