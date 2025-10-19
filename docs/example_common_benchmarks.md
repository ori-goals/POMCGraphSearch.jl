## POMCGS for Common Benchmarks

This section provides recommended settings for the benchmarks reported in the POMCGS paper.  
Users can further tune parameters.
For example, by increasing `max_planning_secs` or `num_sim_per_sa`, or decreasing `max_b_gap`, to derive an offline policy with higher performance.
Reference computation time for each domain is also provided given an modern i7 CPU.


---

### RockSample(7,8)

```julia
using POMCGraphSearch
using POMDPs
using RockSample

pomdp = RockSamplePOMDP(7, 8)

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.2,                # belief merging threshold
    max_search_depth = 40,          # maximum search depth
    num_sim_per_sa = 30             # simulations per action per state particle
)

fsc = solve(pomcgs, pomdp)
run_batch_simulations(pomdp, fsc; n_simulations=10000)
```

POMCGS typically solves **RockSample(7,8)** within **1–5 minutes** (depending on hardware).  
The resulting policy achieves near-optimal performance, comparable to SARSOP on the same instance.

---

### RockSample(11,11)

```julia
pomdp = RockSamplePOMDP(11, 11)

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.3,
    max_search_depth = 40,
    num_sim_per_sa = 20
)

fsc = solve(pomcgs, pomdp)
run_batch_simulations(pomdp, fsc; n_simulations=10000)
```

For **RockSample(11,11)**, POMCGS generally obtains a good offline policy (lower bound comparable to online planners, e.g., value ≈ 12–17) in **under 1 hour**.  Longer runtime may further improve performance (note that the default timeout is 10,000 seconds).

---

### RockSample(15,15)

```julia
pomdp = RockSamplePOMDP(15, 15)

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.4,
    max_search_depth = 40,
    num_sim_per_sa = 20,
    max_planning_secs = 36000.0,
    nb_particles = 50000,
    nb_sim_VMDP = 50000,
    epsilon_VMDP = 0.01
)

fsc = solve(pomcgs, pomdp)
run_batch_simulations(pomdp, fsc; n_simulations=10000)
```

For **RockSample(15,15)**, POMCGS typically obtains a good offline policy(for example, lower bound value ≈ 10–15) within **6 to 9 hours** of computation. We recommend using at least **64 GB of RAM**, as this problem is large scale and memory intensive.


---

### LightDark

```julia
using POMCGraphSearch, POMDPs, POMDPModels

pomdp = LightDark1D()

pomcgs = SolverPOMCGS(pomdp;
    max_b_gap = 0.15,
    state_grid = [1.0, 1.0],       # discretization for continuous states
    num_fixed_observations = 20,   # number of observation clusters
    max_search_depth = 30,
    num_sim_per_sa = 1000
)

fsc = solve(pomcgs, pomdp)
run_batch_simulations(pomdp, fsc; n_simulations=10000)
```

For **LightDark**, POMCGS typically reaches a lower bound value > 3.0 within **10–30 seconds**.  
Note: `state_grid` is required for continuous-state POMDPs to perform belief discretization.

---

### Lidar Roomba

```julia
using POMCGraphSearch, POMDPs, RoombaPOMDPs

num_x_pts, num_y_pts, num_th_pts = 25, 16, 10
sspace = DiscreteRoombaStateSpace(num_x_pts, num_y_pts, num_th_pts)

# Define a large discrete action set for continuous action approximation
max_speed, speed_interval = 5.0, 0.2
max_turn_rate, turn_rate_interval = 1.0, 0.2
action_space = vec([RoombaAct(v, ω)
                    for v in 0:speed_interval:max_speed,
                        ω in -max_turn_rate:turn_rate_interval:max_turn_rate])

pomdp = RoombaPOMDP(sensor=Lidar(),
    mdp=RoombaMDP(config=3, aspace=action_space,
    v_max=max_speed, sspace=sspace))

pomcgs = SolverPOMCGS(pomdp;
    max_search_depth = 40,
    max_b_gap = 0.2,
    bool_APW = true,
    num_sim_per_sa = 200,
    num_fixed_observations = 10
)

fsc = solve(pomcgs, pomdp)
run_batch_simulations(pomdp, fsc; n_simulations=10000)
```
Note that rand(actions(pomdp)) is not properly implemented in the current RoombaPOMDP.jl package.
In this example, we use a large number of discrete actions (286 actions) and then apply action progressive widening (APW) on this action space.
For **Lidar Roomba**, with APW enabled,  POMCGS often reaches a good lower bound (0.5~1.0) in about 2–3 hours.

---

### Bumper Roomba

```julia
using POMCGraphSearch, POMDPs, RoombaPOMDPs

num_x_pts, num_y_pts, num_th_pts = 41, 26, 20
sspace = DiscreteRoombaStateSpace(num_x_pts, num_y_pts, num_th_pts)

max_speed, speed_interval = 5.0, 0.2
max_turn_rate, turn_rate_interval = 1.0, 0.2
action_space = vec([RoombaAct(v, ω)
                    for v in 0:speed_interval:max_speed,
                        ω in -max_turn_rate:turn_rate_interval:max_turn_rate])

pomdp = RoombaPOMDP(sensor=Bumper(),
    mdp=RoombaMDP(config=3, aspace=action_space,
    v_max=max_speed, sspace=sspace))

pomcgs = SolverPOMCGS(pomdp;
    max_search_depth = 60,
    max_b_gap = 0.05,
    bool_APW = true,
    num_sim_per_sa = 100,
    C_star = 1000,
    nb_particles = 100000,
    nb_sim_VMDP = 50000,
    max_planning_secs = 36000.0
)

fsc = solve(pomcgs, pomdp)
run_batch_simulations(pomdp, fsc; n_simulations=10000)
```
**Bumper Roomba** is a much harder problem than **Lidar Roomba**, as the robot relies only on a bumper sensor. For **Bumper Roomba**, with APW enabled, POMCGS generally finds a complete offline policy that outperforms online planners within **6 to 8 hours** of computation.


---

### General Notes
- Default solver settings are designed to be robust and general, while these configurations are recommanded for the listed domains.  
- Hardware and random seeds may cause runtime or minor performance variations.
- This POMCGS implementation is single-threaded.