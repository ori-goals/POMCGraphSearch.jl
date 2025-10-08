mutable struct ContinuousPlanner
	_nb_process_action_samples::Int64            
	_max_b_gap::Float64                          
	_max_graph_node_size::Int64                  
	_nb_iter::Int64                              
	_discount::Float64                           
	_epsilon::Float64                            
	_C_star::Int64								 
	_max_search_depth::Int64					
	_max_planning_secs::Int64                   
	_nb_sim::Int64								 
	_nb_eval::Int64                              
	_Q_learning_policy::Qlearning
	_Log_result::LogResult
	_R_lower::Float64
	_ratio_heuristic_Q::Float64
	_bool_continuous_states::Bool
	_bool_continuous_observations::Bool
	_map_discrete2continuous_states::Dict{Vector{Float64}, Vector{Any}}
	_state_grid::Vector{Float64}
	_num_obs_clusters::Int
	_obs_cluster_model::Any
	_k_a::Float64
    _alpha_a::Float64
    _bool_APW::Bool
end







function ProcessActionWeightedParticle(pomdp,
										fsc::FSC,
										nI::Int64,
										a,
										nb_process_samples::Int64,
										discount::Float64,
										Q_learning_policy::Qlearning,
										ratio_heuristic_Q::Float64,
										bool_continuous_states::Bool,
										bool_continuous_observations::Bool,
										state_grid::Vector{Float64},
										map_discrete2continuous_states::Dict{Vector{Float64}, Vector{Any}},
										obs_cluster_model::Any)


	sum_R_a, sum_all_weights, all_oI_weight, all_dict_weighted_samples = CollectSamplesAndBuildNewBeliefsWeightedParticles(pomdp,
																														fsc,
																														nI,
																														a,
																														nb_process_samples,
																														bool_continuous_states,
																														bool_continuous_observations,
																														state_grid,
																														map_discrete2continuous_states,
																														obs_cluster_model)
												
	# Build new belief nodes
	fsc._nodes[nI]._R_action[a] = sum_R_a
	expected_future_V = 0.0


	merged_belief_for_unexpected_obs = merge_and_normalize_beliefs(all_dict_weighted_samples)

	# find unexpected observations
	for o in fsc._observation_space
		if !haskey(all_dict_weighted_samples, o)
			# for unexpected observations, link to merged belief (belief update with only the action)
			all_dict_weighted_samples[o] = merged_belief_for_unexpected_obs
			all_oI_weight[o] = 0.0
		end
	end

	# for each new belief, check distances to existing belief nodes, and create new nodes if needed
	for (key, value) in all_dict_weighted_samples
		NormalizeDict(all_dict_weighted_samples[key])
		sort!(all_dict_weighted_samples[key], rev = true, byvalue = true)
     	heuristic_value, action_space, heuristic_Q_actions = GetValueQMDP(all_dict_weighted_samples[key], 
																		Q_learning_policy, 
																		pomdp,
																		bool_continuous_states,
																		map_discrete2continuous_states,
																		state_grid)
		bool_search, n_nextI = SearchOrInsertBelief(fsc, all_dict_weighted_samples[key], heuristic_value, fsc._max_accept_belief_gap)
        if !bool_search
            max_Q = HeuristicNodeQ(fsc._nodes[n_nextI], heuristic_Q_actions, ratio_heuristic_Q)
			fsc._nodes[n_nextI]._V_node = max_Q
		end
		fsc._eta[nI][Pair(a, key)] = n_nextI
		obs_weight = all_oI_weight[key]
		expected_future_V += (obs_weight / sum_all_weights) * fsc._nodes[n_nextI]._V_node
	end

	# --- Update Q(n, a) -----
	fsc._nodes[nI]._Q_action[a] = fsc._nodes[nI]._R_action[a] + discount * expected_future_V
	return fsc._nodes[nI]._Q_action[a]
end


function Simulate(pomdp,
	fsc::FSC,
	s,
	nI::Int64,
	depth::Int64,
	max_depth::Int64,
	nb_process_action_samples::Int64,
	discount::Float64,
	C_star::Int64,
	epsilon::Float64,
	Q_learning_policy::Qlearning,
	ratio_heuristic_Q::Float64,
	bool_continuous_states::Bool,
	bool_continuous_observations::Bool,
	state_grid::Vector{Float64},
	map_discrete2continuous_states::Dict{Vector{Float64}, Vector{Any}},
	obs_cluster_model::Any,
	bool_APW::Bool,
	k_a::Float64,
	alpha_a::Float64)

	if depth > max_depth
		return 0
	end


	if (discount^depth) * (Q_learning_policy._R_max - Q_learning_policy._R_min) < epsilon || isterminal(pomdp, s) || nI == -1
		return 0
	end


	if bool_APW
        a = ActionProgressiveWidening(fsc, nI, fsc._action_space, k_a, alpha_a, C_star)
    else
        a = UcbActionSelection(fsc, nI, C_star)
    end


	fsc._nodes[nI]._visits_node += 1
	fsc._nodes[nI]._visits_action[a] += 1

	if fsc._nodes[nI]._visits_action[a] == 1
		return ProcessActionWeightedParticle(pomdp, 
											fsc, 
											nI, 
											a, 
											nb_process_action_samples, 
											discount, 
											Q_learning_policy, 
											ratio_heuristic_Q,
											bool_continuous_states,
											bool_continuous_observations,
											state_grid,
											map_discrete2continuous_states,
											obs_cluster_model)
	end

	nI_next = -1
	sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)

	if bool_continuous_observations
		o_vec = convert_o(Vector{Float64}, o, pomdp)
		o_processed = predict_cluster(obs_cluster_model, o_vec)
		o = o_processed
	end

	if haskey(fsc._eta[nI], Pair(a, o))
		nI_next = fsc._eta[nI][Pair(a, o)]
	else
		nI_next = transition(fsc, nI, a, o)
		fsc._eta[nI][Pair(a, fsc._flag_unexpected_obs)] = nI_next
	end

	# update r(n, a)
	nb_process_samples = nb_process_action_samples
	sum_R_n_a = fsc._nodes[nI]._R_action[a] * (nb_process_samples + fsc._nodes[nI]._visits_action[a])
	sum_R_n_a += r
	fsc._nodes[nI]._R_action[a] = sum_R_n_a / (nb_process_samples + fsc._nodes[nI]._visits_action[a] + 1)

	esti_V = fsc._nodes[nI]._R_action[a] + discount * Simulate(pomdp, 
																fsc, 
																sp, 
																nI_next, 
																depth + 1, 
																max_depth, 
																nb_process_samples, 
																discount, 
																C_star, 
																epsilon, 
																Q_learning_policy, 
																ratio_heuristic_Q, 
																bool_continuous_states,
																bool_continuous_observations, 
																state_grid,
																map_discrete2continuous_states,
																obs_cluster_model,
																bool_APW,
																k_a,
																alpha_a)
	fsc._nodes[nI]._Q_action[a] = fsc._nodes[nI]._Q_action[a] + ((esti_V - fsc._nodes[nI]._Q_action[a]) / fsc._nodes[nI]._visits_action[a])
	fsc._nodes[nI]._V_node = esti_V

	return esti_V
end



function GraphSearchContinuousPOMDP(pomdp,
	b,
	dict_weighted_b::OrderedDict{Any, Float64},
	fsc::FSC,
	planner::ContinuousPlanner)

	# assume an empty fsc
	node_start = CreateNode(b, dict_weighted_b, fsc._action_space)
	heuristic_value, action_space, heuristic_Q_actions = GetValueQMDP(dict_weighted_b, 
																	planner._Q_learning_policy, 
																	pomdp,
																	planner._bool_continuous_states,
																	planner._map_discrete2continuous_states,
																	planner._state_grid)

	HeuristicNodeQ(node_start, heuristic_Q_actions, planner._ratio_heuristic_Q)
	push!(fsc._nodes, node_start)
    push!(fsc._nodes_VQMDP_labels, maximum(values(node_start._Heuristic_Q_action)))

	vec_episodes = Vector{Int64}()
	vec_evaluation_value = Vector{Float64}()
	vec_fsc_size = Vector{Int64}()


    # Headers
    headers = ["Iter", "Total Simulations", "FSC Size", "Lower Bound L", "Upper Bound U", "Planning Time (s)"]
    
    # Print headers with fixed width formatting
    header_string = @sprintf "%6s %18s %12s %15s %15s %18s" headers...
    println(repeat("-", 90))
    println(header_string)
    println(repeat("-", 90))

	sum_planning_time_secs = 0
	for i in 1:planner._nb_iter
		elapsed_time = @elapsed begin
			s = rand(b)
			Simulate(pomdp,
				fsc,
				s,
				1,
				0,
				planner._max_search_depth,
				planner._nb_process_action_samples,
				planner._discount,
				planner._C_star,
				planner._epsilon,
				planner._Q_learning_policy,
				planner._ratio_heuristic_Q,
				planner._bool_continuous_states,
				planner._bool_continuous_observations,
				planner._state_grid,
				planner._map_discrete2continuous_states,
				planner._obs_cluster_model,
				planner._bool_APW,
				planner._k_a,
				planner._alpha_a)
		end

		sum_planning_time_secs += elapsed_time
        
		if sum_planning_time_secs > planner._max_planning_secs
			println("Timeout reached")
			break
		end

		if i % planner._nb_sim == 0

            iter = Int(i ÷ planner._nb_sim)
            fsc_size = length(fsc._nodes)
			U, L = EvaluateBounds(b,
					pomdp, 
					planner._R_lower, 
					fsc, 
					planner._Q_learning_policy, 
					discount(pomdp), 
					planner._nb_eval, 
					planner._C_star,
					planner._epsilon,
					planner._Log_result._vec_evaluation_value,
					planner._Log_result._vec_upper_bound,
					planner._bool_continuous_observations,
					planner._obs_cluster_model)
			

			row_string = @sprintf "%6d %18d %12d %15.6f %15.6f %18.6f" iter i fsc_size L U sum_planning_time_secs
            println(row_string)

			push!(planner._Log_result._vec_episodes, i)
			push!(planner._Log_result._vec_fsc_size, length(fsc._prunned_node_list))
            push!(planner._Log_result._vec_time, sum_planning_time_secs)
			if U - L < planner._epsilon
				break
			end
		end
	end
	

	return vec_episodes, vec_evaluation_value, vec_fsc_size

end



function CollectSamplesAndBuildNewBeliefsWeightedParticles(pomdp,
	fsc::FSC,
	nI::Int64,
	a,
	nb_process_action_samples::Int64,
	bool_continuous_states::Bool,
	bool_continuous_observations::Bool,
	state_grid::Vector{Float64},
	map_discrete2continuous_states::Dict{Vector{Float64}, Vector{Any}},
	obs_cluster_model::Any)

	all_oI_weight = Dict{Int64, Float64}()
	all_dict_weighted_samples = Dict{Int64, OrderedDict{Any, Float64}}()
	sum_R_a = 0.0
	sum_all_weights = 0.0

	# prepare data for multi-thread computation 
	num_threads = Threads.nthreads()
	all_dict_weighted_samples_threads = Vector{Dict{Int64, OrderedDict{Any, Float64}}}()
	all_oI_weight_threads = Vector{Dict{Int64, Float64}}()
	sum_R_a_threads = zeros(Float64, num_threads)
	sum_all_weights_threads = zeros(Float64, num_threads)
	all_obs = Set{Int64}()
    map_discrete2continuous_states_threads= Vector{Dict{Vector{Float64}, Vector{Any}}}()


	for i in 1:num_threads
		push!(all_dict_weighted_samples_threads, Dict{Int64, OrderedDict{Any, Float64}}())
		push!(all_oI_weight_threads, Dict{Int64, Float64}())
		push!(map_discrete2continuous_states_threads, Dict{Vector{Float64}, Vector{Any}}())
	end

	all_keys = collect(keys(fsc._nodes[nI]._dict_weighted_samples))
	Threads.@threads for i in 1:length(all_keys)
		id_thread = Threads.threadid()
		s = all_keys[i]
		w = fsc._nodes[nI]._dict_weighted_samples[s]
		nb_sim = ceil(w * nb_process_action_samples)
		w = 1.0
		for i in 1:nb_sim
			sp, o, r = Step(pomdp, 
							s, 
							a, 
							bool_continuous_states, 
							bool_continuous_observations,
							map_discrete2continuous_states,
							map_discrete2continuous_states_threads,
							state_grid,
							obs_cluster_model,
							id_thread)

			sum_R_a_threads[id_thread] += r * w
			sum_all_weights_threads[id_thread] += w
			if haskey(all_dict_weighted_samples_threads[id_thread], o)
				all_oI_weight_threads[id_thread][o] += w
				if haskey(all_dict_weighted_samples_threads[id_thread][o], sp)
					all_dict_weighted_samples_threads[id_thread][o][sp] += w
				else
					all_dict_weighted_samples_threads[id_thread][o][sp] = w
				end
			else
				all_dict_weighted_samples_threads[id_thread][o] = OrderedDict{Any, Float64}()
				all_oI_weight_threads[id_thread][o] = w
				all_dict_weighted_samples_threads[id_thread][o][sp] = w
				push!(all_obs, o)
			end
		end
	end
	# merge threads data 
	for id_thread in 1:num_threads
		sum_R_a += sum_R_a_threads[id_thread]
		sum_all_weights += sum_all_weights_threads[id_thread]
		if bool_continuous_states
            for (k,v) in map_discrete2continuous_states_threads[id_thread]
                if !haskey(map_discrete2continuous_states, k)
                    map_discrete2continuous_states[k] = v
                end
            end
        end
	end
	sum_R_a = sum_R_a / sum_all_weights


	# merge threads data 
	for o in all_obs
		all_dict_weighted_samples[o] = OrderedDict{Int64, Float64}()
		all_oI_weight[o] = 0.0
	end

	for id_thread in 1:num_threads
		for o in all_obs
			if haskey(all_dict_weighted_samples_threads[id_thread], o)
				all_dict_weighted_samples[o] = merge(+, all_dict_weighted_samples[o], all_dict_weighted_samples_threads[id_thread][o])
				all_oI_weight[o] += all_oI_weight_threads[id_thread][o]
			end
		end
	end

	return sum_R_a, sum_all_weights, all_oI_weight, all_dict_weighted_samples
end




function Step(pomdp,
    s::Any,
    a::Any,
    bool_continuous_states::Bool,
	bool_continuous_observations::Bool,
    map_discrete2continuous_states::Dict{Vector{Float64}, Vector{Any}},
	map_discrete2continuous_states_threads::Vector{Dict{Vector{Float64}, Vector{Any}}},
    state_grid::Vector{Float64},
	obs_cluster_model::Any,
	id_thread::Int64
)

    if !bool_continuous_states && !bool_continuous_observations
        return @gen(:sp, :o, :r)(pomdp, s, a)
    end

    sp = nothing
    o = nothing
    r = 0.0

    if bool_continuous_states
        s = rand(map_discrete2continuous_states[s])
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        sp_vec = convert_s(Vector{Float64}, sp, pomdp)
        sp_processed = ProcessState(sp_vec, state_grid)

        if haskey(map_discrete2continuous_states_threads[id_thread], sp_processed)
            push!(map_discrete2continuous_states_threads[id_thread][sp_processed], sp)
        else
            map_discrete2continuous_states_threads[id_thread][sp_processed] = [sp]
        end

        sp = sp_processed
    end

    if bool_continuous_observations
        o_vec = convert_o(Vector{Float64}, o, pomdp)
        o_processed = predict_cluster(obs_cluster_model, o_vec)
        o = o_processed
    end

    return sp, o, r
end


function Step(pomdp,
    s::Any,
    a::Any,
    bool_continuous_states::Bool,
    map_discrete2continuous_states::Dict{Vector{Float64}, Vector{Any}},
    state_grid::Vector{Float64},
    min_samples_per_grid::Int=100
)

    if !bool_continuous_states
        return @gen(:sp, :o, :r)(pomdp, s, a)
    end

    sp = nothing
    o = nothing
    r = 0.0

    if bool_continuous_states
        s = rand(map_discrete2continuous_states[s])
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
        sp_vec = convert_s(Vector{Float64}, sp, pomdp)
        sp_processed = ProcessState(sp_vec, state_grid)

		if haskey(map_discrete2continuous_states, sp_processed)
			if length(map_discrete2continuous_states[sp_processed]) < min_samples_per_grid
				push!(map_discrete2continuous_states[sp_processed], sp)
			end
		else
			map_discrete2continuous_states[sp_processed] = [sp]
		end
        

        sp = sp_processed
    end

    return sp, o, r
end

function merge_and_normalize_beliefs(all_dict_weighted_samples::Dict{Int, OrderedDict{Any, Float64}})
    merged = OrderedDict{Any, Float64}()

    # 1. Accumulate weights for each state across all beliefs
    for (_, belief) in all_dict_weighted_samples
        for (s, w) in belief
            merged[s] = get(merged, s, 0.0) + w
        end
    end

    # 2. Renormalize
    total_weight = sum(values(merged))
    if total_weight == 0
        error("Total weight is zero when merging beliefs — check inputs.")
    end

    for (s, w) in merged
        merged[s] = w / total_weight
    end

    return merged
end



