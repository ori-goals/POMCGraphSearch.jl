module POMCGS

using JLD2
using JSON
using Dates
using POMDPs
using Clustering
using OrderedCollections
using Printf
using DataFrames, CSV
using LinearAlgebra

include("Qlearning.jl")
include("Utils.jl")
include("FSC.jl")
include("DiscretePlanner.jl")
include("ContinuousPlanner.jl")


export SolverPOMCGS, Solve, SaveFSCPolicyJSON, SaveFSCPolicyJLD2, ExportLogData

mutable struct SolverPOMCGS{POMDP}
    # --- Parameters for the problem model ---
    pomdp::POMDP
    b0::Any
    b0_processed::OrderedDict{Any, Float64}
    max_num_particles::Int
	min_num_particles::Int
	b0_particles::Vector{Any} # initial belief particles
    action_space_type::Symbol
    action_space::Any
    num_init_APW_actions::Int
    num_action_APW_threshold::Int
    state_space_type::Symbol
    state_space::Any
    observation_space_type::Symbol
    observation_space::Any
    num_fixed_observations::Int
    dim_state::Int
    dim_observation::Int
    obs_cluster_model::Any
    num_state_cluster::Int
	state_cluster_model::Any
	state_grid::Vector{Float64}
    # --- Parameters for the VMDP heuristic ---
    VMDP_heuristic::Any
	nb_episode_size::Int
	nb_max_episode::Int
    nb_samples_VMDP::Int
	nb_sim_VMDP::Int
    epsilon_VMDP::Float64
    ratio_heuristic_Q::Float64
    # --- Parameters for the POMCGS planner ---
    nb_process_action_samples::Int64
    max_b_gap::Float64
    max_graph_node_size::Int64
    nb_iter::Int64
    discount::Float64
    epsilon::Float64
    C_star::Int64
    bool_PCA::Bool 
    out_dimension_PCA::Int
    kmeans_itr::Int64
    k_a::Float64
    alpha_a::Float64
    bool_APW::Bool
    max_search_depth::Int64
    max_planning_secs::Int64
    nb_sim::Int64
    nb_eval::Int64
    Log_result::LogResult
    R_lower::Float64
	# --- FSC ---
	fsc::FSC
	# --- Planner ---
	planner::Any

    function SolverPOMCGS(pomdp::POMDP;
                    # --- Problem model defaults ---
                    num_state_cluster::Int64 = 100, # default 100
					state_grid::Vector{Float64} = Vector{Float64}(),
                    num_init_APW_actions::Int = 10, # default number of init fixed actions for continuous action spaces
                    num_action_APW_threshold::Int = 30, # if the action space is larger then this value, use APW
                    num_fixed_observations::Int = 20,
                    max_num_particles::Int = 10_0000,
					min_num_particles::Int = 1_0000,
					b0_particles::Vector{Any} = [], # initial belief particles
                    # --- VMDP heuristic defaults ---
                    # VMDP_heuristic::Any = nothing,
					nb_episode_size::Int = 30,
					nb_max_episode::Int = 20,
                    nb_samples_VMDP::Int = 5000,
					nb_sim_VMDP::Int = 10,
                    epsilon_VMDP::Float64 = 0.01,
                    ratio_heuristic_Q::Float64 = 0.3, # ratio of heuristic Q value in FSC node initialization, if 0, no heuristic Q value (pessimistic), if 1, full heuristic Q value (optimistic)
                    # --- Planner defaults ---
                    nb_process_action_samples::Int64 = 1000,
                    max_b_gap::Float64 = 0.1,
                    max_graph_node_size::Int64 = 10_000_000,
                    nb_iter::Int64 = 10_000_000,
                    epsilon::Float64 = 0.1,
                    C_star::Int64 = 100,
                    # PCA is default disabled
                    bool_PCA::Bool  = false,
                    out_dimension_PCA::Int = 2,
                    kmeans_itr::Int64 = 30,
                    # Action Progressive Widening
                    k_a::Float64 = 2.0,
                    alpha_a::Float64 = 0.2,
                    bool_APW::Bool = false,
                    # Search Parameters
                    max_search_depth::Int64 = 50,
                    max_planning_secs::Int64 = 100_000,
					nb_sim::Int64 = 1000,
                    nb_eval::Int64 = 100_00
                    ) where {POMDP}


        # Detect spaces
        action_space_type, action_space = detect_action_space(pomdp, num_action_APW_threshold, num_init_APW_actions)
        state_space_type, state_space = detect_state_space(pomdp)
        observation_space_type, observation_space = detect_observation_space(pomdp)

        # Initial belief and dimensions
        b0 = initialstate(pomdp)
        sp, o, _ = @gen(:sp, :o, :r)(pomdp, rand(b0), rand(action_space))
        dim_state = length(convert_s(Vector{Float32}, sp, pomdp))
        dim_observation = length(convert_o(Vector{Float32}, o, pomdp))


		b0_particles = b0_particles == [] ? [rand(b0) for _ in 1:min_num_particles] : b0_particles

		# initialize VMDP with Q_learning_Policy
		Q_table = Dict{Any, Dict{Int64, Float64}}()
		V_table = Dict{Any, Float64}()
		learning_rate = 0.9
		explore_rate = 0.7

		Vmdp = Qlearning(Q_table, V_table, learning_rate, explore_rate, action_space, typemin(Float64), typemax(Float64))


        # print details of problem types and parameters
        println("----- POMCGS Initialization -----")
        println("Input POMDP action space type: ", action_space_type)
        println("Input POMDP state space type: ", state_space_type)
        println("Input POMDP observation space type: ", observation_space_type)
        println("Discount factor: ", discount(pomdp))
        
        println("--- Initializing VMDP heuristic ---")
        # Print details of VMDP initialization parameters
        println("VMDP heuristic: Q-learning")
        println("Number of max episodes: ", nb_max_episode)

		# if state is discrete
		if state_space_type == :discrete
            TrainingParallelEpisodes(Vmdp, nb_episode_size, nb_max_episode, nb_samples_VMDP, nb_sim_VMDP, epsilon_VMDP, b0_particles, pomdp)

		else
            TrainingParallelEpisodes(Vmdp, nb_episode_size, nb_max_episode, nb_samples_VMDP, nb_sim_VMDP, epsilon_VMDP, b0_particles, pomdp; use_grid = true, grid_state = state_grid)
        end


		VMDP_heuristic = Vmdp
        # Log result
        log_result = LogResult(Int64[], Float64[], Float64[], Int64[], Float64[])

        # Lower bound on reward
        R_lower = FindRLower(pomdp, b0, action_space)


        # Initialize FSC
		fsc = InitFSC(max_b_gap, max_graph_node_size, action_space, observation_space)

		# init map_discrete2continuous_states
    	map_d2continuous_states = Dict{Vector{Float64}, Vector{Any}}()


		b0_processed = OrderedDict{Any,Float64}()

		if state_space_type == :discrete
			for s in b0_particles
				if haskey(b0_processed, s)
					b0_processed[s] += 1.0/min_num_particles
				else
					b0_processed[s] = 1.0/min_num_particles
				end
			end
		else
			for s in b0_particles
				s_vec = convert_s(Vector{Float64}, s, pomdp)
				s_processed = ProcessState(s_vec, state_grid)
				if haskey(b0_processed, s_processed)
					b0_processed[s_processed] += 1.0/min_num_particles
					push!(map_d2continuous_states[s_processed], s)
				else
					b0_processed[s_processed] = 1.0/min_num_particles
					map_d2continuous_states[s_processed] = [s]
				end
			end

		end

		# if continuous states, discretize the state space (or by state grids)
		state_cluster_model = nothing
        # if continuous observations, discretize the observation space
        obs_cluster_model = nothing

        if observation_space_type == :continuous
            if num_fixed_observations < 2
                throw(ArgumentError("For continuous observation space, num_fixed_observations must be at least 2 for clustering."))
            end
			map_d2continuous_states, obs_clusters, obs_cluster_model = GetMap2RawStatesAndObsClusters(pomdp, 
                                                                                    b0, 
                                                                                    action_space, 
                                                                                    state_space_type, 
                                                                                    num_fixed_observations, 
                                                                                    Vmdp; 
                                                                                    state_grid = state_grid, 
                                                                                    num_trajectories = nb_sim,
                                                                                    trajectory_length = max_search_depth,
                                                                                    map_discrete2continuous_states = map_d2continuous_states)
            println("Observation clustering complete: $num_fixed_observations clusters created.")
            observation_space = [i for i in 1:length(obs_clusters)]

            fsc._observation_space = observation_space
            mu = obs_cluster_model.centers
            obs_kmeans_centroids = Vector{Vector{Float64}}()
            for i in 1:num_fixed_observations
                push!(obs_kmeans_centroids, mu[:,i])
            end

            fsc._obs_kmeans_centroids = obs_kmeans_centroids
        end

		# Init planner
		planner = nothing
		# if everything is discrete, use DiscretePlanner
		if action_space_type == :discrete && state_space_type == :discrete && observation_space_type == :discrete
            planner = DiscretePlanner(nb_process_action_samples, 
											max_b_gap, 
											max_graph_node_size, 
											nb_iter, 
											discount(pomdp),
											epsilon, 
											C_star, 
											max_search_depth, 
											max_planning_secs, 
											nb_sim, 
											nb_eval, 
											VMDP_heuristic,
											log_result,
											R_lower,
                                            ratio_heuristic_Q)
        else 
            # Init planner
            bool_continuous_states = false
            if length(state_grid) > 0 
                bool_continuous_states = true
            end
            
            bool_continuous_observations = false
            if observation_space_type != :discrete
                bool_continuous_observations = true
            end

            if action_space_type != :discrete
                bool_APW = true
            end
            
            planner = ContinuousPlanner(nb_process_action_samples, 
                                        max_b_gap, 
                                        max_graph_node_size, 
                                        nb_iter, 
                                        discount(pomdp),
                                        epsilon, 
                                        C_star, 
                                        max_search_depth, 
                                        max_planning_secs, 
                                        nb_sim, 
                                        nb_eval, 
                                        VMDP_heuristic,
                                        log_result,
                                        R_lower,
                                        ratio_heuristic_Q,
                                        bool_continuous_states,
                                        bool_continuous_observations,
                                        map_d2continuous_states,
                                        state_grid,
                                        num_fixed_observations,
                                        obs_cluster_model,
                                        k_a,
                                        alpha_a,
                                        bool_APW)
        end

        return new{POMDP}(pomdp, b0, b0_processed, max_num_particles, min_num_particles, b0_particles,
                          action_space_type, action_space, num_init_APW_actions, num_action_APW_threshold,
                          state_space_type, state_space,
                          observation_space_type, observation_space, num_fixed_observations,
                          dim_state, dim_observation, obs_cluster_model,
                          num_state_cluster, state_cluster_model, state_grid,
                          VMDP_heuristic, nb_episode_size, nb_max_episode, nb_samples_VMDP, nb_sim_VMDP, epsilon_VMDP, ratio_heuristic_Q,
                          nb_process_action_samples, max_b_gap, max_graph_node_size, nb_iter,
                          discount(pomdp), epsilon, C_star, bool_PCA, out_dimension_PCA, kmeans_itr, k_a, alpha_a, bool_APW,      
                          max_search_depth, max_planning_secs, nb_sim, nb_eval, log_result, R_lower, fsc, planner)
    end
end



function detect_action_space(pomdp, num_action_APW_threshold::Int, num_init_APW_actions::Int)
    if hasmethod(POMDPs.actions, Tuple{typeof(pomdp)})
        acts = POMDPs.actions(pomdp)

        # Check if length is defined → discrete space
        if hasmethod(length, Tuple{typeof(acts)})
            num_actions = length(acts)
            if num_actions > num_action_APW_threshold
                println("Action number ($num_actions) is large, applying action progressive widening (APW")
                bool_APW = true
                action_space = [rand(acts) for _ in 1:num_init_APW_actions]
                return :continuous_sampleable, action_space
            end
            return :discrete, acts
        end

        # Otherwise, check if rand is defined on actions
        if hasmethod(rand, Tuple{typeof(acts)})
            action_space = [rand(acts) for _ in 1:num_fixed_actions]
            return :continuous_sampleable, action_space
        else
            throw(ArgumentError("The POMDP does not have rand function defined for action sampling."))        
        end
    else
        throw(ArgumentError("The POMDP does not have a defined action space."))
    end
end

function detect_state_space(pomdp)
    if hasmethod(POMDPs.states, Tuple{typeof(pomdp)})
        sts = POMDPs.states(pomdp)
        if hasmethod(length, Tuple{typeof(sts)})
            return :discrete, sts
        else
            return :continuous, sts
        end
    else
        return :continuous, nothing
    end
end

function detect_observation_space(pomdp)
    if hasmethod(POMDPs.observations, Tuple{typeof(pomdp)})
        obss = POMDPs.observations(pomdp)
        if hasmethod(length, Tuple{typeof(obss)})
            return :discrete, obss
        else
            return :continuous, obss
        end
    else
        return :continuous, nothing
    end
end


function kmeans_clustering_function(data::AbstractMatrix{<:AbstractFloat}, num_clusters::Int; maxiter::Int=50)
    result = kmeans(data, num_clusters; maxiter = maxiter)
    return result
end


# Function to predict the cluster label for new data points
function predict_cluster(model::M, s::AbstractVector{T}) where {M, T<:AbstractFloat}
    # Find the nearest centroid for each point
    return argmin([norm(s - centroid) for centroid in eachcol(model.centers)])
end


function GetMap2RawStatesAndObsClusters(
    pomdp,
    b0,
    action_space,
    state_space_type::Symbol,      # :discrete or :continuous
    num_obs_clusters::Int,
    Vmdp::Qlearning;
    state_grid::Vector{Float64} = Vector{Float64}(),
    map_discrete2continuous_states = Dict{Vector{Float64}, Vector{Any}}(),
    num_trajectories::Int = 100,
    trajectory_length::Int = 50
)


    #Collect continuous observation samples
    obs_samples = Vector{Vector{Float64}}()

    for traj in 1:num_trajectories
        # Sample initial state from belief b0
        s = rand(b0)

        for t in 1:trajectory_length

            if isterminal(pomdp, s)
                break
            end

            # Choose action depending on state space type
            if state_space_type == :continuous
                if isempty(state_grid)
                    error("State grid is empty. Please provide a valid state grid for continuous states.")
                end
                # Convert continuous state to vector
                s_vec = convert_s(Vector{Float64}, s, pomdp)
                # Map to discretized grid (for Q-learning policy)
                s_processed = ProcessState(s_vec, state_grid)
                a = ChooseActionQlearning(Vmdp, s_processed)

                if haskey(map_discrete2continuous_states, s_processed)
                    push!(map_discrete2continuous_states[s_processed], s)
                else
                    map_discrete2continuous_states[s_processed] = [s]
                end

            else
                a = ChooseActionQlearning(Vmdp, s)
            end

            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            # Collect observation sample
            obs_vec = convert_o(Vector{Float64}, o, pomdp)
            push!(obs_samples, obs_vec)

            s = sp  # advance to next state
        end
    end

    if isempty(obs_samples)
        error("No observation samples collected. Check POMDP's convert_o or simulator.")
    end

    # Convert samples to matrix

    obs_matrix = hcat(obs_samples...)  # each observation is a column

    # Run k-means clustering
    @info "Running k-means clustering on $(size(obs_matrix, 2)) observation samples..."
    kmeans_result = kmeans(obs_matrix, num_obs_clusters; maxiter = 100)

    # Extract cluster centers
    obs_clusters = [kmeans_result.centers[:, i] for i in 1:size(kmeans_result.centers, 2)]

    @info "Observation clustering complete: $num_obs_clusters clusters created."

    return map_discrete2continuous_states, obs_clusters, kmeans_result
end

function Solve(pomcgs::SolverPOMCGS)
    println("----- POMCGS Planning -----")
    println("Initial belief particle size:", length(pomcgs.b0_particles))
    println("Evaluation simulation number:", pomcgs.nb_eval)
    println("Max search depth:", pomcgs.max_search_depth)
    # Run planner
    if pomcgs.planner === nothing
        throw(ArgumentError("No planner is defined for POMCGS. Please reinitialize POMCGS."))
    elseif pomcgs.planner isa DiscretePlanner
        println("Using DiscretePlanner for POMCGS...")
        GraphSearchDiscretePOMDP(
            pomcgs.pomdp,
            pomcgs.b0_particles,
            pomcgs.b0_processed,
            pomcgs.fsc,
            pomcgs.planner
        )
    else
        println("Using ContinuousPlanner for POMCGS...")
        GraphSearchContinuousPOMDP(
            pomcgs.pomdp,
            pomcgs.b0_particles,
            pomcgs.b0_processed,
            pomcgs.fsc,
            pomcgs.planner
        )
    end

    println("--- Planning finished ---")
    println("Total planning time (secs): ", last(pomcgs.planner._Log_result._vec_time))
    pomcgs.fsc._prunned_node_list = Prunning(pomcgs.fsc; MIN_VISITS = pomcgs.planner._C_star) # soft prunning, nodes are not removed from fsc._nodes
    println("FSC size after prunning: ", length(pomcgs.fsc._prunned_node_list))
    println("FSC lower bound value:", last(pomcgs.planner._Log_result._vec_evaluation_value))
end


function SaveFSCPolicyJLD2(pomcgs::SolverPOMCGS; outfile_name::Union{Nothing, String}=nothing)
    # Decide output filename
    filename = if outfile_name === nothing
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        "fsc_$(timestamp).jld2"
    else
        "fsc_$(outfile_name)_result.jld2"
    end

    # Save FSC
    fsc = pomcgs.fsc
    @save filename fsc
    println("FSC result saved to $filename")
end



function SaveFSCPolicyJSON(fsc::FSC; outfile_name::Union{Nothing, String}=nothing, export_obs_clusters::Bool=true)
    # Decide output filename
    filename = if outfile_name === nothing
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
        "fsc_$(timestamp).json"
    else
        "fsc_$(outfile_name)_result.json"
    end

    num_nodes = length(fsc._nodes)
    nodes_json = Vector{Dict{String,Any}}()

    for n in 1:num_nodes
        node = fsc._nodes[n]

        # Collect eta transitions for this node
        eta_entries = []
        for (pair, next_n) in fsc._eta[n]
            push!(eta_entries, Dict(
                "action" => string(pair.first),
                "observation" => pair.second,
                "next_node" => next_n
            ))
        end

        node_dict = Dict(
            "id" => n,
            "best_action" => string(node._best_action),
            "eta" => eta_entries,
            "visits" => node._visits_node,
            "value" => node._V_node
        )

        push!(nodes_json, node_dict)
    end

    fsc_json = Dict(
        "num_nodes" => num_nodes,
        "nodes" => nodes_json
    )

    # --- Write FSC JSON file (pretty print) ---
    open(filename, "w") do io
        JSON.print(io, fsc_json, 4)  # 4 spaces for indentation
    end
    @info "FSC exported to JSON: $filename"

    # --- Export observation cluster centroids if they exist ---
    if export_obs_clusters && !isempty(fsc._obs_kmeans_centroids)
        base = splitext(filename)[1]
        cluster_file = base * "_obs_clusters.json"

        obs_clusters_json = Dict(
            "num_clusters" => length(fsc._obs_kmeans_centroids),
            "clusters" => fsc._obs_kmeans_centroids
        )

        open(cluster_file, "w") do io
            JSON.print(io, obs_clusters_json, 4)  # 4 spaces for indentation
        end

        @info "Observation cluster centroids exported to: $cluster_file"
    end

    return filename
end




end