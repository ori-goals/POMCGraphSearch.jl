mutable struct LogResult
	_vec_episodes::Vector{Int64}
    _vec_lower_bound::Vector{Float64}
	_vec_evaluation_value::Vector{Float64}
    _vec_evaluation_std::Vector{Float64}
	_vec_upper_bound::Vector{Float64}
	_vec_fsc_size::Vector{Int64}
    _vec_time::Vector{Float64}
end

function ExportLogData(pomcgs::Solver, name::String) where {Solver}
    output_name = name *".csv"

    planner = pomcgs.planner

    min_length = min(length(planner._Log_result._vec_episodes),
                    # length(planner._Log_result._vec_lower_bound),
                    # length(planner._Log_result._vec_evaluation_std),
                    length(planner._Log_result._vec_evaluation_value),
                    length(planner._Log_result._vec_upper_bound),
                    length(planner._Log_result._vec_fsc_size),
                    length(planner._Log_result._vec_time))


    df = DataFrame(episode = planner._Log_result._vec_episodes[1:min_length],
                    # lower = planner._Log_result._vec_lower_bound[1:min_length],
                    # eval_std = planner._Log_result._vec_evaluation_std[1:min_length],
                   eval_mean = planner._Log_result._vec_evaluation_value[1:min_length],
                   upper = planner._Log_result._vec_upper_bound[1:min_length],
                   fsc_size = planner._Log_result._vec_fsc_size[1:min_length],
                    time = planner._Log_result._vec_time[1:min_length])
    CSV.write(output_name, string.(df))
end


function NormalizeDict(d::OrderedDict{Int, Float64})
	sum = 0.0
	for (key, value) in d
		sum += value
	end

	for (key, value) in d
		d[key] = value / sum
	end
end

function merge_and_normalize_beliefs(all_dict_weighted_samples::Dict{O, OrderedDict{Int, Float64}}) where {O}
    merged = OrderedDict{Int, Float64}()

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


function ProcessState(s_vec::Vector{Float64}, state_grid::Vector{Float64})
    result = Vector{Int64}()
    for i in 1:length(s_vec)
        s_i = floor(Int64, s_vec[i] / state_grid[i])
        push!(result, s_i)
    end
    return result
end

function detect_action_space(pomdp::POMDP, num_action_APW_threshold::Int, num_init_APW_actions::Int, bool_APW::Bool) where {POMDP}
    try
        if !hasmethod(POMDPs.actions, Tuple{typeof(pomdp)})
            throw(ArgumentError("The POMDP does not have a defined action space."))
        end
        
        acts = POMDPs.actions(pomdp)
        ASpace = typeof(acts)

        # Check if length is defined → discrete space
        if hasmethod(length, Tuple{typeof(acts)})
            num_actions = length(acts)
            if num_actions > num_action_APW_threshold
                println("Action number ($num_actions) is large, applying action progressive widening (APW)")
                bool_APW = true
                action_space = get_uniform_actions(acts, num_init_APW_actions)
                return :continuous_sampleable, ASpace, action_space
            end
            return :discrete, ASpace, acts, false
        end

        # Otherwise, check if rand is defined on actions
        if hasmethod(rand, Tuple{typeof(acts)})
            action_space = get_uniform_actions(acts, num_init_APW_actions)
            return :continuous_sampleable, ASpace, action_space
        else
            throw(ArgumentError("The POMDP does not have rand function defined for action sampling."))        
        end
        
    catch e
        # Handle specific error cases
        error_msg = string(e)
        if occursin("action", lowercase(error_msg)) || occursin("actions", lowercase(error_msg))
            @warn "POMDP action space detection issue: $error_msg"
            # Try to fallback to a default action space if possible
            if hasmethod(POMDPs.actions, Tuple{POMDP})
                # Try the type instead of instance
                acts = POMDPs.actions(POMDP)
                return :discrete, typeof(acts), acts, false
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end
end

function detect_state_space(pomdp::POMDP) where {POMDP}
    try
        if !hasmethod(POMDPs.states, Tuple{typeof(pomdp)})
            return :continuous, Nothing, nothing
        end
        
        sts = POMDPs.states(pomdp)
        SSpace = typeof(sts)
        
        if hasmethod(length, Tuple{typeof(sts)}) && isfinite(length(sts))
            return :discrete, SSpace, sts
        else
            return :continuous, SSpace, sts
        end
        
    catch e
        # Handle state space detection errors
        error_msg = string(e)
        if occursin("state", lowercase(error_msg)) || occursin("states", lowercase(error_msg)) ||
           occursin("continuous", lowercase(error_msg)) || occursin("discrete", lowercase(error_msg))
            
            @warn "POMDP state space detected via error message: $error_msg"
            
            if occursin("continuous", lowercase(error_msg))
                return :continuous, Nothing, nothing
            else
                # Default to continuous for safety
                return :continuous, Nothing, nothing
            end
        else
            @warn "Unexpected error detecting state space: $e. Assuming continuous."
            return :continuous, Nothing, nothing
        end
    end
end

function detect_observation_space(pomdp::POMDP) where {POMDP}
    try
        if hasmethod(POMDPs.observations, Tuple{typeof(pomdp)})
            obss = POMDPs.observations(pomdp)
            OSpace = typeof(obss)
            if hasmethod(length, Tuple{typeof(obss)})
                return :discrete, OSpace, obss
            else
                return :continuous, Vector{Int}, Vector{Int}() # Use Int as placeholder for continuous obs
            end
        else
            return :continuous, Vector{Int}, Vector{Int}()
        end
    catch e
        # If observations() method exists but throws an error (like LidarPOMDP),
        # assume it's continuous observation space
        if occursin("continuous", string(e)) || occursin("Continuous", string(e))
            @warn "POMDP indicates continuous observations via error: $e"
            return :continuous, Vector{Int}, Vector{Int}()
        else
            # Re-throw unexpected errors
            rethrow(e)
        end
    end
end


function get_uniform_actions(acts, num_actions::Int)
    try
        # Strategy 1: Check if it's a finite discrete set
        if hasmethod(length, Tuple{typeof(acts)}) && length(acts) < 10000
            act_list = collect(acts)
            n_total = length(act_list)
            
            if n_total <= num_actions
                return act_list
            else
                # Sample evenly spaced indices
                step = n_total / num_actions
                indices = [round(Int, 1 + (i-1) * step) for i in 1:num_actions]
                return act_list[indices]
            end
        end
        
        # Strategy 2: Check if it's a numeric range
        if hasmethod(minimum, Tuple{typeof(acts)}) && hasmethod(maximum, Tuple{typeof(acts)})
            min_a = minimum(acts)
            max_a = maximum(acts)
            return range(min_a, max_a, length=num_actions) |> collect
        end
        
        # Strategy 3: Check if it's a Julia range
        if acts isa AbstractRange
            return range(first(acts), last(acts), length=num_actions) |> collect
        end
        
        # Strategy 4: Fallback to diverse random sampling
        return get_diverse_random_actions(acts, num_actions)
        
    catch e
        # Final fallback: simple random sampling
        @warn "Failed to get uniform actions, using random sampling: $e"
        return [rand(acts) for _ in 1:num_actions]
    end
end

function get_diverse_random_actions(acts, num_actions::Int; max_attempts::Int=1000)
    # Try to get diverse actions through multiple random samples
    actions = Set()
    attempts = 0
    
    while length(actions) < num_actions && attempts < max_attempts
        push!(actions, rand(acts))
        attempts += 1
    end
    
    return collect(actions)[1:min(num_actions, length(actions))]
end

# Function to predict the cluster label for new data points
function predict_cluster(centers::Matrix{Float64}, s::Vector{Float64})
    # Find the nearest centroid for each point
    return argmin([norm(s - centroid) for centroid in eachcol(centers)])
end


function GetMap2RawStatesAndObsClusters_Weighted_Random(
    pomdp::POMDP,
    action_space::ASpace,
    num_obs_clusters::Int;
    num_trajectories::Int = 1000,
    trajectory_length::Int = 50,
    replication_scale::Float64 = 20.0
) where {POMDP, ASpace}

    # ========== 1. Sampling ==========
    obs_samples = Vector{Vector{Float64}}()
    reward_values = Float64[]
    
    b0 = initialstate(pomdp)
    
    @info "Collecting observation samples from random policy..."
    
    for traj in 1:num_trajectories
        s = rand(b0)
        for t in 1:trajectory_length
            if POMDPs.isterminal(pomdp, s)
                break
            end
            
            a = rand(action_space)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            obs_vec = convert_o(Vector{Float64}, o, pomdp)
            
            push!(obs_samples, obs_vec)
            push!(reward_values, r)
            
            s = sp
        end
    end
    
    if isempty(obs_samples)
        error("No observation samples collected. Check POMDP's convert_o or simulator.")
    end
    
    n_samples = length(obs_samples)
    @info "Collected $n_samples observations"
    
    # ========== 2. Find max reward observations==========
    r_max = maximum(reward_values)
    r_min = minimum(reward_values)
    
    max_indices = findall(reward_values .== r_max)
    
    @info "Max reward: $r_max, found $(length(max_indices)) samples with this reward"
    
    # ========== 3. Save max reward observations ==========
    if isempty(max_indices)
        error("No samples with max reward found!")
    end
    
    max_obs = [obs_samples[i] for i in max_indices]
    
    if length(max_obs) >= 2
        max_obs_matrix = hcat(max_obs...)
        extreme_center = mean(max_obs_matrix, dims=2)[:]
        @info "Using mean of $(length(max_obs)) max-reward observations as extreme cluster center"
    else
        extreme_center = max_obs[1][:]
        @info "Using single max-reward observation as extreme cluster center"
    end
    
    @info "Extreme cluster center (first 5 dims): $(extreme_center[1:min(5, length(extreme_center))])"
    
    # ========== 4. Other observations ==========
    other_indices = findall(reward_values .< r_max)
    
    if isempty(other_indices)
        @warn "All observations have the same reward ($r_max). Returning single cluster."
        dummy_kmeans = (centers = hcat(extreme_center),
                       totalcost = 0.0,
                       assignments = ones(Int, length(obs_samples)),
                       counts = [length(obs_samples)])
        return [extreme_center], dummy_kmeans
    end
    
    other_obs = [obs_samples[i] for i in other_indices]
    other_rewards = [reward_values[i] for i in other_indices]
    
    @info "Other samples: $(length(other_obs)) with rewards in [$r_min, $r_max)"
    
    # ========== 5. Compute other observation weights ==========
    if r_max > r_min
        other_normalized = (other_rewards .- r_min) ./ (r_max - r_min)
        distance_to_max = 1.0 .- other_normalized
        weights = distance_to_max .^ 2 .+ 0.1
        weights = weights ./ mean(weights)
    else
        weights = ones(length(other_obs))
    end
    
    @info "Weight statistics: min=$(minimum(weights)), max=$(maximum(weights)), mean=$(mean(weights))"
    
    # ========== 6. Replicate ==========
    function replicate_by_weight(X::Matrix{Float64}, w::Vector{Float64}, scale::Float64)
        cols = Vector{Vector{Float64}}()
        N = size(X, 2)
        total = 0
        for j in 1:N
            count = max(1, Int(round(w[j] * scale)))
            total += count
            for _ in 1:count
                push!(cols, X[:, j])
            end
        end
        @info "Replicated $N samples to $total samples"
        return hcat(cols...)
    end
    
    other_matrix = hcat(other_obs...)
    other_weighted = replicate_by_weight(other_matrix, weights, replication_scale)
    
    # ========== 7. Clustering ==========
    remaining_clusters = max(1, num_obs_clusters - 1)
    actual_k = min(remaining_clusters, size(other_weighted, 2))
    
    if actual_k >= 2
        @info "Clustering remaining observations into $actual_k clusters..."
        kmeans_other = kmeans(other_weighted, actual_k; maxiter=100)
        other_centers = [kmeans_other.centers[:, i] for i in 1:size(kmeans_other.centers, 2)]
    elseif actual_k == 1
        @info "Only one remaining cluster, using mean"
        other_centers = [mean(other_weighted, dims=2)[:]]
        kmeans_other = (centers = hcat(other_centers[1]),
                       totalcost = 0.0,
                       assignments = ones(Int, size(other_weighted, 2)),
                       counts = [size(other_weighted, 2)])
    else
        error("No remaining observations to cluster!")
    end
    
    # ========== 8. Combine all clusters ==========
    all_centers = [extreme_center]
    append!(all_centers, other_centers)
    
    @info "Final clustering: $(length(all_centers)) clusters (1 max-reward cluster + $(length(other_centers)) normal clusters)"
    
    # ========== 9. Build kmeans_result ==========
    full_centers = hcat([c for c in all_centers]...)
    
    all_obs_matrix = hcat(obs_samples...)
    distances = [sum((all_obs_matrix[:, i] .- full_centers[:, j]) .^ 2) 
                 for i in 1:size(all_obs_matrix, 2), j in 1:size(full_centers, 2)]
    assignments = [argmin(distances[i, :]) for i in 1:size(distances, 1)]
    
    counts = [count(x -> x == j, assignments) for j in 1:size(full_centers, 2)]
    
    totalcost = sum([sum((all_obs_matrix[:, i] .- full_centers[:, assignments[i]]) .^ 2) 
                     for i in 1:size(all_obs_matrix, 2)])
    
    kmeans_result = (centers = full_centers,
                    totalcost = totalcost,
                    assignments = assignments,
                    counts = counts)
    
    # ========== 10. Print ==========
    @info "Cluster sizes: $(counts)"
    
    return all_centers, kmeans_result
end



function generate_initial_particles(b0::B, num_particles::Int) where {B}
    # --- Case 1: Dict or OrderedDict
    if b0 isa AbstractDict
        states = collect(keys(b0))
        probs = collect(values(b0))
        probs ./= sum(probs)
        counts = round.(Int, probs .* num_particles)
        
        # Adjust rounding error to make total exact
        total = sum(counts)
        if total != num_particles
            diff = num_particles - total
            # assign extras to largest probs
            order = sortperm(probs, rev=true)
            for i in 1:abs(diff)
                idx = order[mod1(i, length(order))]
                counts[idx] += sign(diff)
            end
        end
        
        # Expand into particles
        return vcat([fill(states[i], counts[i]) for i in eachindex(states)]...)
    end

    # --- Case 2: SparseCat
    if b0 isa SparseCat
        vals, probs = b0.vals, b0.probs
        probs ./= sum(probs)
        counts = round.(Int, probs .* num_particles)
        total = sum(counts)
        if total != num_particles
            diff = num_particles - total
            order = sortperm(probs, rev=true)
            for i in 1:abs(diff)
                idx = order[mod1(i, length(order))]
                counts[idx] += sign(diff)
            end
        end
        return vcat([fill(vals[i], counts[i]) for i in eachindex(vals)]...)
    end

    # --- Case 3: Vector of states (uniform)
    if b0 isa AbstractVector
        n = length(b0)
        full_repeats = div(num_particles, n)
        remainder = mod(num_particles, n)
        return vcat([b0 for _ in 1:full_repeats]..., b0[1:remainder])
    end


    # --- Case 4: Generic with rand(b0)
    if hasmethod(rand, (typeof(b0),))
        particles = [rand(b0) for _ in 1:num_particles]  # sample particles from the initial belief
        return particles
    end

    throw(ArgumentError("Unsupported belief type $(typeof(b0))"))
end




function sample_key_from_weighted_dict(dict::OrderedDict{Int, Float64})
    # Extract keys and weights
    keys_vec = collect(keys(dict))
    weights_vec = collect(values(dict))
    
    # Sample one key based on weights
    sampled_key = sample(keys_vec, Weights(weights_vec))
    
    return sampled_key
end