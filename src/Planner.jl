mutable struct Planner
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
	_ratio_heuristic_Q::Float64
	_k_a::Float64
    _alpha_a::Float64
    _bool_APW::Bool
end



function ProcessActionWeightedParticle(model::Model,
										fsc::FSC,
										nI::Int64,
										a::A,
										discount::Float64,
										Q_learning_policy::Qlearning,
										ratio_heuristic_Q::Float64
										) where {A}


	sum_R_a, sum_all_weights, all_oI_weight, all_dict_weighted_samples = CollectSamplesAndBuildNewBeliefsWeightedParticles(model::Model,
																														fsc,
																														nI,
																														a)
												
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
																		model)
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


function Simulate(model::Model,
				fsc::FSC,
				s::Int,
				nI::Int64,
				depth::Int64,
				max_depth::Int64,
				discount::Float64,
				C_star::Int64,
				epsilon::Float64,
				Q_learning_policy::Qlearning,
				ratio_heuristic_Q::Float64,
				bool_APW::Bool,
				k_a::Float64,
				alpha_a::Float64)

	if depth > max_depth
		return 0
	end


	if (discount^depth) * (Q_learning_policy._R_max - Q_learning_policy._R_min) < epsilon || isterminal(model, s) || nI == -1
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
		return ProcessActionWeightedParticle(model, 
											fsc, 
											nI, 
											a, 
											discount, 
											Q_learning_policy, 
											ratio_heuristic_Q)
	end

	nI_next = -1
	sp, o, r = Step(model, s, a)

	if haskey(fsc._eta[nI], Pair(a, o))
		nI_next = fsc._eta[nI][Pair(a, o)]
	else
		nI_next = transition(fsc, nI, a, o)
		fsc._eta[nI][Pair(a, fsc._flag_unexpected_obs)] = nI_next
	end


	esti_V = fsc._nodes[nI]._R_action[a] + discount * Simulate(model, 
																fsc, 
																sp, 
																nI_next, 
																depth + 1, 
																max_depth, 
																discount, 
																C_star, 
																epsilon, 
																Q_learning_policy, 
																ratio_heuristic_Q, 
																bool_APW,
																k_a,
																alpha_a)

	fsc._nodes[nI]._Q_action[a] = fsc._nodes[nI]._Q_action[a] + ((esti_V - fsc._nodes[nI]._Q_action[a]) / fsc._nodes[nI]._visits_action[a])
	fsc._nodes[nI]._V_node = esti_V

	return esti_V
end



function MCGraphSearchPOMDP(model::Model,
							b::Vector{Int},
							dict_weighted_b::OrderedDict{Int, Float64},
							fsc::FSC,
							planner::Planner)

	pomdp = model.pomdp
	b0 = initialstate(pomdp)

	# assume an empty fsc
	node_start = CreateNode(dict_weighted_b, fsc._action_space)
	heuristic_value, action_space, heuristic_Q_actions = GetValueQMDP(dict_weighted_b, 
																	planner._Q_learning_policy, 
																	model)

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
			Simulate(model,
				fsc,
				s,
				1,
				0,
				planner._max_search_depth,
				planner._discount,
				planner._C_star,
				planner._epsilon,
				planner._Q_learning_policy,
				planner._ratio_heuristic_Q,
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
			U, L = EvaluateBounds(pomdp, 
									fsc, 
									planner._Q_learning_policy, 
									POMDPs.discount(pomdp), 
									planner._nb_eval, 
									planner._C_star,
									planner._epsilon,
									planner._Log_result._vec_evaluation_value,
									planner._Log_result._vec_upper_bound)
			

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


function CollectSamplesAndBuildNewBeliefsWeightedParticles(
    model::Model,
    fsc::FSC,
    nI::Int64,
    a::A
) where {A}
    node = fsc._nodes[nI]
    weighted_particles = node._dict_weighted_samples

    all_oI_weight = Dict{Int64, Float64}()
    all_dict_weighted_samples = Dict{Int64, OrderedDict{Int, Float64}}()

    sum_R_a = 0.0
    sum_all_weights = 0.0

    # Iterate directly over weighted particles
    for (s, w) in weighted_particles
        steps = Step_batch(model, s, a)

        @inbounds for (sp, o, r) in steps
            sum_R_a += r * w
            sum_all_weights += w

            # Retrieve or initialize observation dictionary
            odict = get!(all_dict_weighted_samples, o, OrderedDict{Int, Float64}())
            all_oI_weight[o] = get(all_oI_weight, o, 0.0) + w

            # Accumulate next-state weight
            odict[sp] = get(odict, sp, 0.0) + w
        end
    end

    # Normalize reward safely
    sum_R_a = (sum_all_weights > 0) ? (sum_R_a / sum_all_weights) : 0.0

    return sum_R_a, sum_all_weights, all_oI_weight, all_dict_weighted_samples
end






