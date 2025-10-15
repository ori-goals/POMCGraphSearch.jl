mutable struct FscNode{A}
    _Q_action::Dict{A,Float64}
    _Heuristic_Q_action::Dict{A,Float64}
    _R_action::Dict{A,Float64} # expected instant reward 
    _visits_action::Dict{A,Int64}
    _visits_node::Int64
    _V_node::Float64
    _best_action::A
    _dict_weighted_samples::OrderedDict{Int,Float64} # state index -> weight, state index is handled by ModelWrapper
    _actions::Vector{A} # for continuous action space, store the sampled actions
end


mutable struct FSC{A, O, ASpace, OSpace} <: Policy
    _eta::Vector{Dict{Pair{A,O},Int64}}
    _nodes_VQMDP_labels::Vector{Float64}
    _obs_kmeans_centroids::Matrix{Float64}
    _nodes::Vector{FscNode}
    _max_accept_belief_gap::Float64
    _max_node_size::Int64
    _action_space::ASpace
    _observation_space::OSpace
    _prunned_node_list::Vector{Int64}
end

# Belief updater for FSC - tracks current node as belief
struct FSCBeliefUpdater{A, O, ASpace, OSpace, P} <: Updater
    fsc::FSC{A, O, ASpace, OSpace}
    pomdp::P  # Store POMDP for observation conversion if needed
end


# Get action from FSC node
function POMDPs.action(fsc::FSC, node_id::Int)
    # return GetBestAction(fsc._nodes[node_id])
    return fsc._nodes[node_id]._best_action
end

# Get updater for this policy
function POMDPs.updater(fsc::FSC, pomdp::POMDP)
    return FSCBeliefUpdater(fsc, pomdp)
end

# Update belief (node) based on action and observation
function POMDPs.update(updater::FSCBeliefUpdater, current_node::Int, a::A, observation::O) where {A, O}
    # Warn if action is not the node's best action
    # if action != updater.fsc._nodes[current_node]._best_action
    #     @warn "Action $action is not the best action for node $current_node"
    # end
    
    # Handle different observation types
    if updater.fsc._obs_kmeans_centroids != zeros(Float64, 0, 0)
        # Continuous observation: convert to discrete using clustering
        return transition_continuous(updater.fsc, current_node, a, observation, updater.pomdp)
    else
        # Discrete observation: use directly
        return transition(updater.fsc, current_node, a, observation)
    end
end

# Initialize belief to starting node
function POMDPs.initialize_belief(updater::FSCBeliefUpdater, initial_state_distribution::Any)
    return 1  # Start from node 1
end

function run_standard_simulation(pomdp::POMDP, fsc::FSC; 
                                max_steps::Int=100, 
                                verbose::Bool=false,
                                initial_node::Int=1)
    
    history = simulate(HistoryRecorder(max_steps=max_steps), 
                      pomdp, fsc, updater(fsc, pomdp), initial_node)
    
    if verbose
        println("Simulation finished after $(length(history)) steps.")
        println("Total discounted reward: $(discounted_reward(history))")
        
        for (i, (s, a, r, sp, o, b)) in enumerate(eachstep(history, "s,a,r,sp,o,b"))
            println("Step $i: Node $b, State $s, Action $a, Reward $r, Obs $o")
        end
    end
    
    return history
end

function run_batch_simulations(pomdp::POMDP, fsc::FSC;
                             n_simulations::Int=1000,
                             max_steps::Int=100,
                             initial_node::Int=1,
                             verbose::Bool=false)
    
    rewards = Float64[]
    
    for i in 1:n_simulations
        history = run_standard_simulation(pomdp, fsc, 
                                        max_steps=max_steps,
                                        initial_node=initial_node,
                                        verbose=false)
        reward = discounted_reward(history)
        push!(rewards, reward)
        
        if verbose && i % 100 == 0
            println("Completed $i/$n_simulations simulations")
        end
    end
    
    println("Batch Evaluation Results:")
    println("Mean reward: $(mean(rewards))")
    println("Std dev: $(std(rewards))")
    
    return mean(rewards), std(rewards)
end
# ------------------ FSC methods ------------------

function InitFscNode(action_space::ASpace) where {ASpace}
    A = eltype(action_space)
    best_action = first(action_space)

    init_actions = []
    # --- init for actions ---
    init_Q_action = Dict{A,Float64}()
    init_heuristic_Q_action = Dict{A,Float64}()
    init_R_action = Dict{A,Float64}()
    init_visits_action = Dict{A,Int64}()
    for a in action_space
        init_Q_action[a] = 0.0
        init_heuristic_Q_action[a] = 0.0
        init_R_action[a] = 0.0
        init_visits_action[a] = 0
        push!(init_actions, a)
    end
    # ------------------------
    init_visits_node = 0
    init_V_node = 0.0
    # --- Weighted Particles ----
    init_dict_weighted_particles = OrderedDict{Int,Float64}()
    return FscNode{A}(init_Q_action,
                    init_heuristic_Q_action,
                    init_R_action,
                    init_visits_action,
                    init_visits_node,
                    init_V_node,
                    best_action,
                    init_dict_weighted_particles,
                    init_actions)
end

function CreateNode(weighted_b::OrderedDict{Int, Float64}, action_space::ASpace) where {ASpace}
    node = InitFscNode(action_space)
    node._dict_weighted_samples = weighted_b
    return node
end


function InitFSC(max_accept_belief_gap::Float64, max_node_size::Int64, action_space::ASpace, observation_space::OSpace) where {ASpace, OSpace}
    A = eltype(action_space)
    O = eltype(observation_space)
    init_eta = Vector{Dict{Pair{A,O},Int64}}(undef, max_node_size)
    for i in range(1, stop=max_node_size)
        init_eta[i] = Dict{Pair{A,O},Int64}()
    end
    init_nodes_VQMDP_labels = Vector{Float64}() # node index -> VQMDP label
    init_obs_kmeans_centroids = Matrix{Float64}(undef, 0, 0) # each column is a centroid
    init_nodes = Vector{FscNode}()
    init_prunned_node_list = Vector{Int64}()

    return FSC(init_eta,
        init_nodes_VQMDP_labels,
        init_obs_kmeans_centroids,
        init_nodes,
        max_accept_belief_gap,
        max_node_size,
        action_space,
        observation_space,
        init_prunned_node_list
       )

end

function GetBestAction(n::FscNode)
    Q_max = typemin(Float64)
    best_a = first(keys(n._Q_action))
    visits = n._visits_action
    q_actions = n._Q_action
    
    @inbounds for (a, q_value) in q_actions
        if visits[a] > 0 && q_value > Q_max
            Q_max = q_value
            best_a = a
        end
    end
    
    n._best_action = best_a
    return best_a
end


function UcbActionSelection(fsc::FSC, nI::Int64, C_star::Int64)
    node_visits = fsc._nodes[nI]._visits_node
    max_value = typemin(Float64)
    current_max_value, selected_a = findmax(fsc._nodes[nI]._Q_action)


    if node_visits > C_star
        return selected_a
    end


    for a in fsc._action_space
        ratio_visit = 0
        node_a_visits = fsc._nodes[nI]._visits_action[a]

        c = (fsc._nodes[nI]._Heuristic_Q_action[a] - fsc._nodes[nI]._Q_action[a])
        if node_a_visits == 0
            ratio_visit = log(node_visits + 1) / 0.1
        else
            ratio_visit = log(node_visits + 1) / node_a_visits
        end

        value = fsc._nodes[nI]._Q_action[a] + c * sqrt(ratio_visit)

        if value > max_value
            max_value = value
            selected_a = a
        end

    end

    return selected_a
end

function ActionProgressiveWidening(fsc::FSC, nI::Int, action_space, K_a::Float64, alpha_a::Float64, C_star::Int64)
    node_visits = fsc._nodes[nI]._visits_node
    current_action_num = length(fsc._nodes[nI]._actions)
    if current_action_num <= K_a*(node_visits^alpha_a) && node_visits < C_star
        a = rand(action_space)
        AddNewAction(fsc._nodes[nI], a)
        return a
    else
        return UcbActionSelection(fsc, nI, C_star) 
    end
end

function AddNewAction(n::FscNode, a::A) where {A}
    if !haskey(n._visits_action, a)
        push!(n._actions, a)
        n._abstract_observations[a] = Vector{Vector{Float64}}()
        n._Q_action[a] = 0.0
        n._R_action[a] = 0.0
        n._visits_action[a] = 0.0
    end
end

function SearchSimilarBelief(
    fsc::FSC,
    node_list::Vector{Int64},
    new_weighted_particles::OrderedDict{Int, Float64},
    b_gap_max::Float64
)
    min_distance_node_i = -1
    min_distance = typemax(Float64)

    # Pre-extract keys and values for faster access
    new_keys = collect(keys(new_weighted_particles))
    new_values = collect(values(new_weighted_particles))

    for node_i in node_list
        weighted_b_node_i = fsc._nodes[node_i]._dict_weighted_samples

        distance_i = 0.0

        # Fast loop through belief particles
        @inbounds for j in eachindex(new_keys)
            key = new_keys[j]
            value = new_values[j]
            distance_i += abs(value - get(weighted_b_node_i, key, 0.0))
            if distance_i > b_gap_max
                # Early stop: no need to continue
                break
            end
        end

        if distance_i < min_distance
            min_distance = distance_i
            min_distance_node_i = node_i
        end
    end

    return min_distance, min_distance_node_i
end


function SearchOrInsertBelief(
    fsc::FSC,
    new_weighted_particles::OrderedDict{Int,Float64},
    new_heuristic_value::Float64,
    b_gap_max::Float64;
    Kcandidates::Int = 1000
)
    N = length(fsc._nodes)

    # Precompute heuristic values for all nodes
    heuristic_values = fsc._nodes_VQMDP_labels

    # Pick top-K nodes closest in heuristic value
    diffs = abs.(heuristic_values .- new_heuristic_value)
    sorted_idx = sortperm(diffs)
    candidate_idxs = sorted_idx[1:min(Kcandidates, N)]

    min_distance, min_node_idx = SearchSimilarBelief(fsc, candidate_idxs, new_weighted_particles, b_gap_max)

    # Insert new node if no close match found
    if min_distance > b_gap_max
        new_node = CreateNode(new_weighted_particles, fsc._action_space)
        push!(fsc._nodes, new_node)
        push!(fsc._nodes_VQMDP_labels, new_heuristic_value)
        return false, length(fsc._nodes)
    else
        return true, min_node_idx
    end
end


function Prunning(fsc::FSC; MIN_VISITS::Int = 50)
    nI = 1
    open_list = [nI]
    result_list = [nI]
    while !isempty(open_list)
        nI = pop!(open_list)

        if  fsc._nodes[nI]._visits_node >= MIN_VISITS
            # Reliable best action: follow policy pruning
            a_best = GetBestAction(fsc._nodes[nI])
                for (k, v) in fsc._eta[nI]
                if k[1] == a_best && !(v in result_list)
                    push!(open_list, v)
                    push!(result_list, v)
                end
            end
        else
            # Underexplored node: keep all children to avoid accidental pruning
            for (_, v) in fsc._eta[nI]
                if !(v in result_list)
                    push!(open_list, v)
                    push!(result_list, v)
                end
            end
        end
    end

    fsc._prunned_node_list = result_list
    return result_list
end


function EvaluateBounds(
    pomdp::POMDP,
    fsc::FSC{A, O, ASpace, OSpace},
    Q_learning_policy::Qlearning{A},
    discount::Float64,
    nb_sim::Int,
    C_star::Int,
    epsilon::Float64,
    vec_evaluation_value::Vector{Float64},
    vec_upper_bound::Vector{Float64}
) where {POMDP, A, O, ASpace, OSpace}

    obs_cluster_model = fsc._obs_kmeans_centroids
    bool_continuous_observations = length(obs_cluster_model) > 0

    b0 = initialstate(pomdp)
    R_max = Q_learning_policy._R_max
    R_min = Q_learning_policy._R_min

    sum_r_U::Float64 = 0.0
    sum_r_L::Float64 = 0.0

    # Preallocate vectors to avoid reallocations
    o_vec = Vector{Float64}()

    @inbounds for _ in 1:nb_sim
        s = rand(b0)
        nI::Int = 1
        step::Int = 0

        while (discount^step) * (R_max - R_min) > epsilon && !POMDPs.isterminal(pomdp, s)
            a = GetBestAction(fsc._nodes[nI])::A
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)

            if bool_continuous_observations
                empty!(o_vec)
                o_vec = convert_o(Vector{Float64}, o, pomdp)
                o = predict_cluster(obs_cluster_model, o_vec)::Int
            end

            if haskey(fsc._eta[nI], Pair(a, o)) && fsc._nodes[nI]._visits_node > C_star
                nI = fsc._eta[nI][Pair(a, o)]
                sum_r_U += (discount^step) * r
                sum_r_L += (discount^step) * r
            else
                max_Q = fsc._nodes_VQMDP_labels[nI]
                sum_r_U += (discount^step) * max_Q
                sum_r_L += (discount^step) * SimulationWithFSC(
                                                            pomdp,
                                                            s,
                                                            fsc;
                                                            discount = discount,
                                                            epsilon = epsilon,
                                                            R_max = R_max,
                                                            R_min = R_min,
                                                            nI_init = nI,
                                                            bool_continuous_observations = bool_continuous_observations,
                                                            obs_cluster_model = obs_cluster_model
                                                        )
                break
            end

            s = sp
            step += 1
        end
    end

    U = sum_r_U / nb_sim
    L = sum_r_L / nb_sim

    push!(vec_upper_bound, U)
    push!(vec_evaluation_value, L)

    return U, L
end


function SimulationWithFSC(pomdp::POMDP,
                            start_state::S,
                            fsc::FSC{A, O, ASpace, OSpace};           
                            max_steps::Int = 100,
                            discount::Float64 = 1.0,
                            epsilon::Float64 = 0.01,
                            R_max::Float64 = 0.0,
                            R_min::Float64 = 0.0,
                            nI_init::Int = 1,
                            bool_continuous_observations::Bool = false,
                            obs_cluster_model::Matrix{Float64} = zeros(Float64, 0, 0),
                            verbose::Bool = false
    ) where {POMDP, A, O, S, ASpace, OSpace} 
    # --- Initialize starting state ---
    s = start_state
    nI = nI_init
    sum_r = 0.0
    step = 0

    while step ≤ max_steps && (discount^step)*(R_max - R_min) > epsilon && POMDPs.isterminal(pomdp, s) == false

        a = GetBestAction(fsc._nodes[nI])::A
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)

        if bool_continuous_observations
            o_vec = convert_o(Vector{Float64}, o, pomdp)
            o = predict_cluster(obs_cluster_model, o_vec)
        end

        sum_r += (discount^step) * r

        if verbose
            println("---------")
            println("Step: ", step)
            println("State: ", s)
            println("Action: ", a)
            println("Observation: ", o)
            println("Reward: ", r)
            println("Node: ", nI)
            println("Node visits: ", fsc._nodes[nI]._visits_node)
            println("Node value: ", fsc._nodes[nI]._V_node)
        end

        s = sp
        nI = transition(fsc, nI, a, o)
        step += 1
    end

    if verbose
        println("Simulation finished after $step steps. Total discounted reward: $sum_r")
    end

    return sum_r
end



function HeuristicNodeQ(node::FscNode, Heuristic_Q_actions::Dict{A, Float64}, ratio::Float64) where {A}
	max_value = typemin(Float64)
	for (a, value) in node._Q_action
		value = 0.0
		if haskey(Heuristic_Q_actions, a)
            value = Heuristic_Q_actions[a]
        end

        node._Heuristic_Q_action[a] = value
        # node._Q_action[a] = ratio*value
        node._Q_action[a] = value

		if value > max_value
			max_value = value
		end
	end
	return ratio*max_value
end


function GetValueQMDP(
    b::OrderedDict{Int, Float64},
    Q_learning_policy::Qlearning{A},
    model::Model
) where {A}

    discount_factor = POMDPs.discount(model.pomdp)
    action_space = Q_learning_policy._action_space

    Q_actions = Dict{A, Float64}(a => 0.0 for a in action_space)
    max_value = -Inf


    states = collect(keys(b))
    beliefs = collect(values(b))
    n_states = length(states)

    for a in action_space
        temp_value = 0.0
        total_weight = 0.0

        # For each belief particle (state s)
        @inbounds for i in 1:n_states
            s = states[i]
            pb = beliefs[i]

            # Get transition distribution 
            transitions = Step_batch(model, s, a)

            # Iterate over all possible transitions with probabilities
            for ((sp, _), (prob, avg_r)) in transitions
                # Use probability-weighted expected value
                # prob: P(sp,o|s,a), avg_r: expected reward
                temp_value += pb * prob * (avg_r + discount_factor * GetV(Q_learning_policy, sp))
                total_weight += pb * prob
            end
        end

        if total_weight > 0
            temp_value /= total_weight  # Normalize
        end

        Q_actions[a] = temp_value
        if temp_value > max_value
            max_value = temp_value
        end
    end

    return max_value, action_space, Q_actions
end


function transition(fsc::FSC, nI::Int, a::A, o::O) where {A, O}
    node_transitions = fsc._eta[nI]  # Dict{Pair{A,O}, Int}
    a_o_pair = Pair(a, o)

    # 1. Exact match for (a, o)
    if haskey(node_transitions, a_o_pair)
        return node_transitions[a_o_pair]
    else
        # 2. Gather candidate next nodes for same action a
        candidates = [next_node for (pair, next_node) in node_transitions if pair.first == a]

        if isempty(candidates)
            # println("Warning: No transitions found for action $a from node $nI with observation $o.")
            return 1  # No transition for this action, go to root node
        end
        throw(ArgumentError("Invalid transition with node $nI, action $a, and observation $o."))        
    end
end

function transition_continuous(fsc::FSC, nI::Int, a::A, o::O_Continuous, pomdp::POMDP) where {A, O_Continuous, POMDP}
    # Convert continuous observation to discrete cluster
    observation_vec = convert_o(Vector{Float64}, o, pomdp)
    obs_discrete = predict_cluster(fsc._obs_kmeans_centroids, observation_vec)::Int
    
    # Use discrete transition
    return transition(fsc, nI, a, obs_discrete)
end