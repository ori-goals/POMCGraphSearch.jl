module POMCGraphSearch

using JLD2
using JSON
using Dates
using POMDPs
using Clustering
using OrderedCollections
using Printf
using DataFrames, CSV
using LinearAlgebra

include("ModelWrapper.jl")
include("Qlearning.jl")
include("Utils.jl")
include("FSC.jl")
include("Planner.jl")


export SolverPOMCGS, Solve, SaveFSCPolicyJSON, SaveFSCPolicyJLD2, ExportLogData

mutable struct SolverPOMCGS{POMDP, ASpace, OSpace_discrete, S, A, O_discrete}
    # --- Parameters for the problem model ---
    pomdp::POMDP
    model::Model{POMDP, S, A, O_discrete}
    b0_processed::OrderedDict{Int, Float64}
    max_num_particles::Int
	nb_particles::Int
    action_space_type::Symbol
    action_space::ASpace
    num_init_APW_actions::Int
    num_action_APW_threshold::Int
    state_space_type::Symbol
    # state_space::SSpace
    observation_space_type::Symbol
    observation_space::OSpace_discrete
    num_fixed_observations::Int
    obs_cluster_model::Matrix{Float64} # kmeans model for observation clustering
    num_sim_per_sa::Int
	state_grid::Vector{Float64}
    # --- Parameters for the VMDP heuristic ---
    VMDP_heuristic::Qlearning{A}
	nb_episode_size::Int
	nb_max_episode::Int
    nb_samples_VMDP::Int
	nb_sim_VMDP::Int
    epsilon_VMDP::Float64
    ratio_heuristic_Q::Float64
    # --- Parameters for the POMCGS planner ---
    max_b_gap::Float64
    max_graph_node_size::Int64
    nb_iter::Int64
    discount::Float64
    epsilon::Float64
    C_star::Int64
    kmeans_itr::Int64
    k_a::Float64
    alpha_a::Float64
    bool_APW::Bool
    max_search_depth::Int64
    max_planning_secs::Int64
    nb_sim_per_iter::Int64
    nb_eval::Int64
    Log_result::LogResult
	# --- FSC ---
	fsc::FSC{A, O_discrete, ASpace, OSpace_discrete}
	# --- Planner ---
	planner::Planner

    function SolverPOMCGS(pomdp::POMDP;
                    # --- Problem model defaults ---
                    num_sim_per_sa::Int64 = 10, # default 10
					state_grid::Vector{Float64} = Vector{Float64}(),
                    num_init_APW_actions::Int = 10, # default number of init fixed actions for continuous action spaces
                    num_action_APW_threshold::Int = 30, # if the action space is larger then this value, use APW
                    num_fixed_observations::Int = 20,
                    max_num_particles::Int = 10_0000,
					nb_particles::Int = 1_0000,
                    # --- VMDP heuristic defaults ---
                    # VMDP_heuristic::Any = nothing,
					nb_episode_size::Int = 30,
					nb_max_episode::Int = 20,
                    nb_samples_VMDP::Int = 5000,
					nb_sim_VMDP::Int = 10,
                    epsilon_VMDP::Float64 = 0.01,
                    ratio_heuristic_Q::Float64 = 0.3, # ratio of heuristic Q value in FSC node initialization, if 0, no heuristic Q value (pessimistic), if 1, full heuristic Q value (optimistic)
                    # --- Planner defaults ---
                    max_b_gap::Float64 = 0.3,
                    max_graph_node_size::Int64 = 10_000_000,
                    nb_iter::Int64 = 10_000_000,
                    epsilon::Float64 = 0.1,
                    C_star::Int64 = 100,
                    kmeans_itr::Int64 = 30,
                    # Action Progressive Widening
                    k_a::Float64 = 2.0,
                    alpha_a::Float64 = 0.2,
                    bool_APW::Bool = false,
                    # Search Parameters
                    max_search_depth::Int64 = 40,
                    max_planning_secs::Int64 = 100_000,
					nb_sim_per_iter::Int64 = 1000,
                    nb_eval::Int64 = 100_00
                    ) where {POMDP}


        # Detect spaces
        action_space_type, ASpace, action_space = detect_action_space(pomdp, num_action_APW_threshold, num_init_APW_actions)
        state_space_type, SSpace, state_space = detect_state_space(pomdp)
        observation_space_type, OSpace_discrete, observation_space = detect_observation_space(pomdp)

        S = statetype(pomdp)
        A = actiontype(pomdp)
        O = obstype(pomdp)
        O_discrete = observation_space_type == :discrete ? O : Int


        # if continuous observations, discretize the observation space
        obs_cluster_model = zeros(Float64, 0, 0)
        bool_continuous_observations = false

        if observation_space_type == :continuous
            bool_continuous_observations = true
            if num_fixed_observations < 2
                throw(ArgumentError("For continuous observation space, num_fixed_observations must be at least 2 for clustering."))
            end

            obs_clusters, kmeans_result =  GetMap2RawStatesAndObsClusters_Weighted_Random(pomdp,
                                                                            action_space, 
                                                                            num_fixed_observations; 
                                                                            num_trajectories = nb_sim_per_iter,
                                                                            trajectory_length = max_search_depth)
                                                                            
            println("Observation clustering complete: $num_fixed_observations clusters created.")
            observation_space = [i for i in 1:length(obs_clusters)]

            obs_cluster_model = kmeans_result.centers


            OSpace = typeof(observation_space)
        end

        # Initialize FSC
		fsc = InitFSC(max_b_gap, max_graph_node_size, action_space, observation_space)
        fsc._obs_kmeans_centroids = obs_cluster_model


        # Init the model wrapper
        model = InitModel(pomdp, num_sim_per_sa, nb_particles; state_grid=state_grid, O_discrete_type = O_discrete, bool_continuous_observations=bool_continuous_observations, obs_cluster_model=obs_cluster_model)

        # Initial belief
        b0 = initialstate(pomdp)  # initial belief state
        b0_particles = model.b0_particles

		# initialize VMDP with Q_learning_Policy
		Q_table = Dict{Int, Dict{A, Float64}}()
		V_table = Dict{Int, Float64}()
		learning_rate = 0.9
		explore_rate = 0.7

		Vmdp = Qlearning(Q_table, V_table, learning_rate, explore_rate, action_space, typemin(Float64), typemax(Float64))


        # print details of problem types and parameters
        println("----- POMCGS Initialization -----")
        println("Input POMDP action space type: ", action_space_type)
        println("Input POMDP state space type: ", state_space_type)
        println("Input POMDP observation space type: ", observation_space_type)
        println("Discount factor: ", POMDPs.discount(pomdp))
        
        println("--- Initializing VMDP heuristic ---")
        # Print details of VMDP initialization parameters
        println("VMDP heuristic: Q-learning")
        println("Number of max episodes: ", nb_max_episode)

		# if state is discrete
        TrainingEpisodes(Vmdp, nb_episode_size, nb_max_episode, nb_samples_VMDP, nb_sim_VMDP, epsilon_VMDP, model)

		VMDP_heuristic = Vmdp
        # Log result
        log_result = LogResult(Int64[], Float64[], Float64[], Int64[], Float64[])

		b0_processed = OrderedDict{Int,Float64}()

        for s in b0_particles
            if haskey(b0_processed, s)
                b0_processed[s] += 1.0/nb_particles
            else
                b0_processed[s] = 1.0/nb_particles
            end
        end





		# Init planner
        bool_continuous_observations = false
        if observation_space_type != :discrete
            bool_continuous_observations = true
        end

        if action_space_type != :discrete
            bool_APW = true
        end
        
        planner = Planner(max_b_gap, 
                            max_graph_node_size, 
                            nb_iter, 
                            POMDPs.discount(pomdp),
                            epsilon, 
                            C_star, 
                            max_search_depth, 
                            max_planning_secs, 
                            nb_sim_per_iter, 
                            nb_eval, 
                            VMDP_heuristic,
                            log_result,
                            ratio_heuristic_Q,
                            k_a,
                            alpha_a,
                            bool_APW) 
                                   
        # end

        return new{POMDP, ASpace, OSpace_discrete, S, A, O_discrete}(pomdp, model, b0_processed, max_num_particles, nb_particles,
                          action_space_type, action_space, num_init_APW_actions, num_action_APW_threshold,
                          state_space_type,
                          observation_space_type, observation_space, num_fixed_observations, obs_cluster_model,
                          num_sim_per_sa, state_grid,
                          VMDP_heuristic, nb_episode_size, nb_max_episode, nb_samples_VMDP, nb_sim_VMDP, epsilon_VMDP, ratio_heuristic_Q,
                          max_b_gap, max_graph_node_size, nb_iter,
                          POMDPs.discount(pomdp), epsilon, C_star, kmeans_itr, k_a, alpha_a, bool_APW,      
                          max_search_depth, max_planning_secs, nb_sim_per_iter, nb_eval, log_result, fsc, planner)
    end
end


function kmeans_clustering_function(data::AbstractMatrix{<:AbstractFloat}, num_clusters::Int; maxiter::Int=50)
    result = kmeans(data, num_clusters; maxiter = maxiter)
    return result
end




function CreateObsClusters(model::Model,
                        action_space,
                        num_obs_clusters::Int,
                        Vmdp::Qlearning;
                        num_trajectories::Int = 100,
                        trajectory_length::Int = 50
)


    #Collect continuous observation samples
    obs_samples = Vector{Vector{Float64}}()

    for traj in 1:num_trajectories
        # Sample initial state from belief b0
        s = rand(model.b0_particles)
        for t in 1:trajectory_length
            if isterminal(model, s)
                break
            end
            a = ChooseActionQlearning(Vmdp, s)
            sp, o, r = Step(model, s, a)
            # Collect observation sample
            obs_vec = convert_o(Vector{Float64}, o, model.pomdp)
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

    return obs_clusters, kmeans_result
end


function Solve(pomcgs::SolverPOMCGS)
    println("----- POMCGS Planning -----")
    println("Initial belief particle size:", length(pomcgs.model.b0_particles))
    println("Evaluation simulation number:", pomcgs.nb_eval)
    println("Max search depth:", pomcgs.max_search_depth)
    # Run planner
    if pomcgs.planner === nothing
        throw(ArgumentError("No planner is defined for POMCGS. Please reinitialize POMCGS."))
    else
        MCGraphSearchPOMDP(
            pomcgs.model,
            pomcgs.model.b0_particles,
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


    obs_kmeans_centroids = Vector{Vector{Float64}}()
    num_fixed_observations = length(fsc._observation_space)
    for i in 1:num_fixed_observations
        push!(obs_kmeans_centroids, fsc._obs_kmeans_centroids[:,i])
    end

    # --- Export observation cluster centroids if they exist ---
    if export_obs_clusters && !isempty(obs_kmeans_centroids)
        base = splitext(filename)[1]
        cluster_file = base * "_obs_clusters.json"

        obs_clusters_json = Dict(
            "num_clusters" => length(obs_kmeans_centroids),
            "clusters" => obs_kmeans_centroids
        )

        open(cluster_file, "w") do io
            JSON.print(io, obs_clusters_json, 4)  # 4 spaces for indentation
        end

        @info "Observation cluster centroids exported to: $cluster_file"
    end

    return filename
end




end