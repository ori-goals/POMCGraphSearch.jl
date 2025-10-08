mutable struct FscNode{S}
    _state_particles::Vector{S} # Vector of particles, each particle is of type S
    _Q_action::Dict{Any,Float64}
    _Heuristic_Q_action::Dict{Any,Float64}
    _R_action::Dict{Any,Float64} # expected instant reward 
    _visits_action::Dict{Any,Int64}
    _visits_node::Int64
    _V_node::Float64
    _best_action::Any
    _dict_weighted_samples::OrderedDict{Any,Float64}
    _actions::Vector{Any} # for continuous action space, store the sampled actions
end


mutable struct FSC
    _eta::Vector{Dict{Pair{Any,Int64},Int64}}
    _nodes_VQMDP_labels::Vector{Float64} # node index -> VQMDP label
    _obs_kmeans_centroids::Vector{Vector{Float64}}
    _nodes::Vector{FscNode}
    _max_accept_belief_gap::Float64
    _max_node_size::Int64
    _action_space::Any
    _observation_space::Any
    _dict_trans_func::Dict{Any,Dict{Any,Dict{Any,Float64}}} # a->s->s'
    _dict_obs_func::Dict{Any,Dict{Any,Dict{Any,Float64}}} # a->s'->o
    _dict_process_a_s::Dict{Pair{Any,Any},Bool} # (a, s) -> bool
    _flag_unexpected_obs::Int64
    _prunned_node_list::Vector{Int64}
    _map_discrete2continuous_states::Dict{Vector{Float64}, Vector{Any}}
end

function InitFscNode(action_space)
    init_actions = []
    init_particles = []
    # --- init for actions ---
    init_Q_action = Dict{Any,Float64}()
    init_heuristic_Q_action = Dict{Any,Float64}()
    init_R_action = Dict{Any,Float64}()
    init_visits_action = Dict{Any,Int64}()
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
    init_dict_weighted_particles = OrderedDict{Any,Float64}()
    return FscNode(init_particles,
        init_Q_action,
        init_heuristic_Q_action,
        init_R_action,
        init_visits_action,
        init_visits_node,
        init_V_node,
        nothing,
        init_dict_weighted_particles,
        init_actions)

end

function CreateNode(b::Vector{S}, weighted_b::OrderedDict{Any, Float64}, action_space) where {S}
    node = InitFscNode(action_space)
    node._state_particles = b
    node._dict_weighted_samples = weighted_b
    return node
end


function InitFSC(max_accept_belief_gap::Float64, max_node_size::Int64, action_space, observation_space)
    init_eta = Vector{Dict{Pair{Any,Int64},Int64}}(undef, max_node_size)
    for i in range(1, stop=max_node_size)
        init_eta[i] = Dict{Pair{Any,Int64},Int64}()
    end
    init_nodes_VQMDP_labels = Vector{Float64}() # node index -> VQMDP label
    init_obs_kmeans_centroids = Vector{Vector{Float64}}() # node index -> QMDP best action label
    init_nodes = Vector{FscNode}()
    init_dict_trans_func = Dict{Any,Dict{Any,Dict{Any,Float64}}}() # a->s->s'
    init_dict_obs_func = Dict{Any,Dict{Any,Dict{Any,Float64}}}() # a->s'->o
    init_dict_process_a_s = Dict{Pair{Any,Any},Bool}() # (a, s) -> bool
    flag_unexpected_obs = -999
    init_prunned_node_list = Vector{Int64}()
    init_map_discrete2continuous_states = Dict{Vector{Float64}, Vector{Any}}()

    return FSC(init_eta,
        init_nodes_VQMDP_labels,
        init_obs_kmeans_centroids,
        init_nodes,
        max_accept_belief_gap,
        max_node_size,
        action_space,
        observation_space,
        init_dict_trans_func,
        init_dict_obs_func,
        init_dict_process_a_s,
        flag_unexpected_obs,
        init_prunned_node_list,
        init_map_discrete2continuous_states)

end

function GetBestAction(n::FscNode)
    Q_max = typemin(Float64)
    best_a = rand(keys(n._Q_action))
    for (key, value) in n._Q_action
        if value > Q_max && n._visits_action[key] != 0
            Q_max = value
            best_a = key
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

function AddNewAction(n::FscNode, a)
    if !haskey(n._visits_action, a)
        push!(n._actions, a)
        n._abstract_observations[a] = Vector{Vector{Float64}}()
        n._Q_action[a] = 0.0
        n._R_action[a] = 0.0
        n._visits_action[a] = 0.0
    end
end

function SearchSimiliarBelief(fsc::FSC, node_list::Vector{Int64}, new_weighted_particles::OrderedDict{Any,Float64}, b_gap_max::Float64)

    min_distance_node_i = -1
    min_distance = typemax(Float64)


    num_threads = Threads.nthreads()
    min_distance_node_i_threads = zeros(Int64, num_threads)
    min_distance_threads = ones(Float64, num_threads) * typemax(Float64)


    Threads.@threads for i in 1:length(node_list)
        node_i = node_list[i]
        id_thread = Threads.threadid()
        distance_i = 0.0
        weighted_b_node_i = fsc._nodes[node_i]._dict_weighted_samples
        for (key, value) in new_weighted_particles
            if haskey(weighted_b_node_i, key)
                distance_i += abs(value - weighted_b_node_i[key])
            else
                distance_i += value
            end

            if distance_i > b_gap_max
                break
            end
        end

        if distance_i < min_distance_threads[id_thread]
            min_distance_threads[id_thread] = distance_i
            min_distance_node_i_threads[id_thread] = node_i
        end
        # end
    end

    for i in 1:num_threads
        if min_distance_threads[i] < min_distance
            min_distance = min_distance_threads[i]
            min_distance_node_i = min_distance_node_i_threads[i]
        end
    end

    return min_distance, min_distance_node_i
end

function SearchOrInsertBelief(
    fsc::FSC,
    new_weighted_particles::OrderedDict{Any,Float64},
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

    min_distance, min_node_idx = SearchSimiliarBelief(fsc, candidate_idxs, new_weighted_particles, b_gap_max)

    # Insert new node if no close match found
    if min_distance > b_gap_max
        new_node = CreateNode([], new_weighted_particles, fsc._action_space)
        push!(fsc._nodes, new_node)
        push!(fsc._nodes_VQMDP_labels, new_heuristic_value)
        return false, length(fsc._nodes)
    else
        return true, min_node_idx
    end
end



function ComputeDistance(vec_1::Vector{Float64}, vec_2::Vector{Float64})
    distance = 0.0
    for i in 1:length(vec_2)
        distance_temp = abs(vec_2[i] - vec_1[i])
        distance += distance_temp
    end
    return distance
end

function ComputeDistance(dict_1::Dict{Any, Float64}, dict_2::Dict{Any, Float64})
    sum = 0.0
    for (key, value) in dict_1
        if haskey(dict_2, key)
            sum += abs(value - dict_2[key])
        else
            sum += value
        end
    end
    
    return sum
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


function EvaluateBounds(b0,
						pomdp, 
						R_lower_bound, 
						fsc::FSC, 
						Q_learning_policy::Qlearning, 
						discount::Float64, 
						nb_sim::Int64, 
						C_star::Int64,
						epsilon::Float64,
						vec_evaluation_value::Vector{Float64},
						vec_upper_bound::Vector{Float64},
						bool_continuous_observations::Bool,
						obs_cluster_model::Any)
    sum_r_U = 0.0
    sum_r_L = 0.0
    R_max = Q_learning_policy._R_max
    R_min = Q_learning_policy._R_min

    num_threads = Threads.nthreads()
    sum_r_U_threads = zeros(Float64, num_threads)
    sum_r_L_threads = zeros(Float64, num_threads)

    Threads.@threads for sim_i = 1:nb_sim
        id_thread = Threads.threadid()        
        step = 0
        s = rand(b0)
        nI = 1

        # while (discount^step)*(R_max - R_min)/( 1 - discount) > epsilon && isterminal(pomdp, s) == false
        while (discount^step)*(R_max - R_min) > epsilon && isterminal(pomdp, s) == false
            a = GetBestAction(fsc._nodes[nI])
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
			if bool_continuous_observations
				o_vec = convert_o(Vector{Float64}, o, pomdp)
				o_processed = predict_cluster(obs_cluster_model, o_vec)
				o = o_processed
			end


            if haskey(fsc._eta[nI], Pair(a, o)) && fsc._nodes[nI]._visits_node > C_star && nI != -1
                nI = fsc._eta[nI][Pair(a, o)]
                sum_r_U_threads[id_thread] += (discount^step) *  r 
                sum_r_L_threads[id_thread] += (discount^step) *  r 
            else
                # need check GetValueQMDP function
                max_Q = fsc._nodes_VQMDP_labels[nI]
                sum_r_U_threads[id_thread] += (discount^step)*max_Q
                sum_r_L_threads[id_thread] += (discount^step)*SimulationWithFSC(pomdp,
                                                                                fsc;
                                                                                start_state = s,
                                                                                discount = discount,
                                                                                epsilon = epsilon,
                                                                                R_max = R_max,
                                                                                R_min = R_min,
                                                                                nI_init = nI,
                                                                                bool_continuous_observations = bool_continuous_observations,
                                                                                obs_cluster_model = obs_cluster_model)
                break
            end

			s = sp

            step += 1
        end

    end

    for i in 1: num_threads
        sum_r_U += sum_r_U_threads[i]
        sum_r_L += sum_r_L_threads[i]
    end

	U = sum_r_U / nb_sim
	L = sum_r_L / nb_sim

	push!(vec_upper_bound, U)
	push!(vec_evaluation_value, L)

    return U, L 
end

function SimulationWithFSC(
    pomdp,
    fsc::FSC;
    b0 = nothing,                        
    start_state = nothing,               
    max_steps::Int = 100,
    discount::Float64 = 1.0,
    epsilon::Float64 = 0.0,
    R_max::Float64 = 0.0,
    R_min::Float64 = 0.0,
    nI_init::Int = 1,
    bool_continuous_observations::Bool = false,
    obs_cluster_model::Any = nothing,
    verbose::Bool = false
)
    # --- Initialize starting state ---
    s = start_state !== nothing ? start_state : rand(b0)
    nI = nI_init
    sum_r = 0.0
    step = 0

    while step ≤ max_steps &&
          (epsilon <= 0.0 || (discount^step) * (R_max - R_min) > epsilon) &&
          !isterminal(pomdp, s) &&
          nI != -1

        a = GetBestAction(fsc._nodes[nI])
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



function HeuristicNodeQ(node::FscNode, Heuristic_Q_actions::Dict{Any, Float64}, ratio::Float64)
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


function GetValueQMDP(b::OrderedDict{Any, Float64}, 
                        Q_learning_policy::Qlearning, 
                        pomdp,
                        bool_continuous_states::Bool,
                        map_d2continuous_states::Union{Nothing, Dict{Vector{Float64}, Vector{Any}}},
                        state_grid::Union{Nothing, Vector{Float64}};
                        nb_sim::Int = 10)
    max_value = typemin(Float64)
    Q_actions = Dict{Any, Float64}() 

    for a in Q_learning_policy._action_space
        temp_value = 0.0
        for (s, pb) in b
            for _ in 1:nb_sim

                if bool_continuous_states
                    sp, o, r = Step(pomdp, 
                    s, 
                    a, 
                    bool_continuous_states, 
                    map_d2continuous_states,
                    state_grid)
                else
                    sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
                end

                temp_value += pb*(r + discount(pomdp)*GetV(Q_learning_policy, sp))
            end
        end
        temp_value /= nb_sim
        Q_actions[a] = temp_value
        if temp_value > max_value
            max_value = temp_value
        end
    end

    return max_value, Q_learning_policy._action_space, Q_actions
end

function transition(fsc::FSC, nI::Int, a::Any, o::Any)
    # Terminal node or invalid index
    if nI == -1
        return -1
    end

    node_transitions = fsc._eta[nI]  # Dict{Pair{Any,Int}, Int}
    a_o_pair = Pair(a, o)

    # 1. Exact match for (a, o)
    if haskey(node_transitions, a_o_pair)
        return node_transitions[a_o_pair]
    else
        # 2. Gather candidate next nodes for same action a
        candidates = [next_node for (pair, next_node) in node_transitions if pair.first == a]

        if isempty(candidates)
            return -1
        end
        throw(ArgumentError("Invalid transition with node $nI, action $a, and observation $o."))        
    end
end