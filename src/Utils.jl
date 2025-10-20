mutable struct LogResult
	_vec_episodes::Vector{Int64}
	_vec_evaluation_value::Vector{Float64}
	_vec_upper_bound::Vector{Float64}
	_vec_fsc_size::Vector{Int64}
    _vec_time::Vector{Float64}
end

function ExportLogData(planner::Planner, name::String) where {Planner}
    output_name = name *".csv"

    min_length = min(length(planner._Log_result._vec_episodes),
                    length(planner._Log_result._vec_evaluation_value),
                    length(planner._Log_result._vec_upper_bound),
                    length(planner._Log_result._vec_valid_value),
                    length(planner._Log_result._vec_unvalid_rate),
                    length(planner._Log_result._vec_fsc_size),
                    length(planner._Log_result._vec_time))


    df = DataFrame(episode = planner._Log_result._vec_episodes[1:min_length],
                   lower = planner._Log_result._vec_evaluation_value[1:min_length],
                   upper = planner._Log_result._vec_upper_bound[1:min_length],
                   valid_value = planner._Log_result._vec_valid_value[1:min_length],
                   unvalid_rate = planner._Log_result._vec_unvalid_rate[1:min_length],
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

function merge_and_normalize_beliefs(all_dict_weighted_samples::Dict{Int, OrderedDict{Int, Float64}})
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

    obs_samples = Vector{Vector{Float64}}()
    reward_weights = Float64[]
    
    # Track reward range
    r_min = Inf
    r_max = -Inf
    total_rewards = 0.0
    reward_count = 0

    b0 = initialstate(pomdp)  # initial belief state

    for traj in 1:num_trajectories
        s = rand(b0)

        for t in 1:trajectory_length
            if POMDPs.isterminal(pomdp, s)
                break
            end

            # --- Use random action ---
            a = rand(action_space)

            # --- Simulate step ---
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
            obs_vec = convert_o(Vector{Float64}, o, pomdp)
            push!(obs_samples, obs_vec)
            
            # Update reward statistics
            r_min = min(r_min, r)
            r_max = max(r_max, r)
            total_rewards += r
            reward_count += 1
            
            # Record raw reward for later weight calculation
            push!(reward_weights, r)

            s = sp
        end
    end

    if isempty(obs_samples)
        error("No observation samples collected. Check POMDP's convert_o or simulator.")
    end

    # Print detailed reward statistics
    r_mean = total_rewards / reward_count
    @info "Reward Statistics from Random Policy Exploration:" *
          "\n  - R_min: $r_min" *
          "\n  - R_max: $r_max" * 
          "\n  - R_mean: $r_mean" *
          "\n  - Total samples: $reward_count" *
          "\n  - Reward range: $(r_max - r_min)"
    
    # Check if exploration is sufficient
    if r_max - r_min < 1e-6
        @warn "Very small reward range detected. Random policy may not be exploring the environment sufficiently."
    elseif r_max == r_min
        @warn "All rewards are identical. Consider increasing exploration or checking environment dynamics."
    end

    # Calculate relative absolute weights
    if r_max ≈ r_min  # All rewards are the same
        relative_weights = ones(length(reward_weights))
        @info "Using uniform weights (all rewards are equal)"
    else
        # Use relative absolute value: |r - r_min| / (r_max - r_min)
        relative_weights = [abs(r - r_min) / (r_max - r_min) for r in reward_weights]
        
        # Ensure minimum weight is not zero
        min_weight = 0.1  # Avoid zero weights
        relative_weights = [max(w, min_weight) for w in relative_weights]
        
        @info "Weight statistics: min=$(minimum(relative_weights)), max=$(maximum(relative_weights)), mean=$(mean(relative_weights))"
    end

    # Convert to matrix
    obs_matrix = hcat(obs_samples...)

    # --- Replication function ---
    function replicate_by_weight(X::Matrix{Float64}, weights::Vector{Float64}, scale::Float64)
        cols = Vector{Vector{Float64}}()
        N = size(X, 2)
        for j in 1:N
            w = max(1, Int(round(weights[j] * scale)))
            for _ in 1:w
                push!(cols, X[:, j])
            end
        end
        return hcat(cols...)
    end

    @info "Replicating observation samples by relative |reward| with scale=$replication_scale..."
    obs_matrix_weighted = replicate_by_weight(obs_matrix, relative_weights, replication_scale)

    @info "Running k-means on $(size(obs_matrix_weighted, 2)) (weighted) observation samples..."
    kmeans_result = kmeans(obs_matrix_weighted, num_obs_clusters; maxiter = 100)

    # Extract cluster centers
    obs_clusters = [kmeans_result.centers[:, i] for i in 1:size(kmeans_result.centers, 2)]

    @info "Observation clustering complete: $num_obs_clusters clusters created (with relative |reward| weighting)."

    return obs_clusters, kmeans_result
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