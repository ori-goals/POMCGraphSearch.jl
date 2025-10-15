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
| `./POMCGraphSearch.jl` | Main POMCGS algorithm |
| `./ModelWrapper.jl` |  Model wrapper for continuous domains |
| `./Planner.jl` | General POMCGS planner implementation |
| `./FSC.jl` | Defines the FSC data structure used during planning |
| `./Qlearning.jl` | Implements the MDP heuristic method |
| `./Utils.jl` | Utility functions |

---

## Getting Started

### Recommended Setup

> **Important**: Before using POMCGraphSearch, we recommend creating a **new Julia environment** to avoid dependency conflicts:

```julia
using Pkg
Pkg.activate("ExamplePOMCGSEnv")  # create a clean environment
Pkg.add("POMCGraphSearch")
```


### Example — RockSample (with parameters)

```julia
using POMCGraphSearch
using POMDPs
using RockSample

pomdp = RockSamplePOMDP(7, 8)

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.2,                # belief merging threshold
    max_search_depth = 30,          # maximum search depth
    num_sim_per_sa = 20 # simulations per action for a given state particle
)

fsc = solve(pomcgs, pomdp)          # Solve using POMDPs.jl API

run_standard_simulation(pomdp, fsc; verbose=true)  # Simulate the resulting FSC
```

---

### Example — LightDark (Continuous POMDP)

```julia
using POMCGraphSearch 

using POMDPModels

pomdp = LightDark1D()  # define the LightDark problem

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.2, 
    state_grid = [1.0, 1.0], # the state grid for state discretization
    num_fixed_observations = 20, # the number of observation clusters
    max_search_depth = 30,
    num_sim_per_sa = 1000
)  # Initialize the POMCGS solver. It will automatically use the continuous planner for this problem.

fsc = solve(pomcgs, pomdp)          # Solve using POMDPs.jl API

run_standard_simulation(pomdp, fsc; verbose=true)  # Simulate the resulting FSC
```
---

### Saving FSC Policies

```Julia
SaveFSCPolicyJSON(pomcgs.fsc) # save the fsc policy to a JSON file
```

or 

```Julia
SaveFSCPolicyJLD2(pomcgs.fsc) # save the fsc policy to a JLD2 file
```

---

### Core POMCGS Planning Parameters

| Parameter                   | Default  | Description                                                                                  |
|------------------------------|---------|----------------------------------------------------------------------------------------------|
| `max_b_gap`                  | `0.1`   | Belief merging threshold; controls granularity of the belief graph. *(0.1 = tight, larger = faster planning but coarser graph)* |
| `max_search_depth`           | `50`    | Maximum search depth for tree search.                                                       |
| `num_sim_per_sa`  | `100`  | Number of simulations per action.                                                           |
| `epsilon`                    | `0.1`   | Algorithm stops when U-L < epsilon (convergence threshold).                                 |
| `nb_particles`          | `10000` | Number of particles sampled from the initial belief `b_0`.                                  |
| `state_grid`                 | `[]`    | Grid for discretizing continuous states (used only for continuous-state POMDPs).           |
| `num_fixed_observations`     | `20`    | Number of observation clusters (used only for continuous-observation POMDPs).              |


### Q-learning Initialization Parameters (for MDP upper bound)

During initialization, POMCGS uses Q-learning to compute an approximate MDP value for the upper bound.  
The default values should work for most POMDPs tested in the paper.  
If initialization is too slow, inaccurate, or gives unexpected results, you can adjust the following parameters:

| Parameter          | Default  | Description |
|-------------------|---------|-------------|
| `nb_episode_size`  | `30`    | Number of steps per episode in the Q-learning initialization. |
| `VMDP_nb_max_episode`   | `20`    | Maximum number of episodes to run during initialization. |
| `nb_samples_VMDP`  | `5000`  | Number of samples used to estimate the MDP value function. |
| `nb_sim_VMDP`      | `10`    | Number of simulations per sample for value estimation. |
| `epsilon_VMDP`     | `0.1`  | Convergence threshold for Q-learning value computation. |


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
