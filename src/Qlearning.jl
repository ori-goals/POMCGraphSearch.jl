mutable struct Qlearning
    _Q_table::Dict{Any, Dict{Any, Float64}} #s -> a -> Q
    _V_table::Dict{Any, Float64} # s -> V
    _learning_rate::Float64
    _explore_rate::Float64
    _action_space
    _R_max::Float64
    _R_min::Float64
end

function ChooseActionQlearning(Q_learning_Policy::Qlearning, s)
    a_selected = -1
    rand_num = rand()
    if rand_num < Q_learning_Policy._explore_rate
        a_selected = rand(Q_learning_Policy._action_space)
    else
        a_selected = BestAction(Q_learning_Policy::Qlearning, s)
    end
    return a_selected
end

function MaxQ(Q_learning_Policy::Qlearning, s)
    max_Q = typemin(Float64)
    for a in Q_learning_Policy._action_space
        Q_temp = GetQ(Q_learning_Policy, s, a)
        if Q_temp > max_Q
            max_Q = Q_temp
        end
    end

    Q_learning_Policy._V_table[s] = max_Q
    return max_Q
end

function GetV(Q_learning_Policy::Qlearning, s)
    if haskey(Q_learning_Policy._V_table, s)
        return Q_learning_Policy._V_table[s]
    else
        return 0.0
    end
end

function BestAction(Q_learning_Policy::Qlearning, s)
    max_Q = typemin(Float64)
    a_max_Q = -1
    for a in Q_learning_Policy._action_space
        Q_temp = GetQ(Q_learning_Policy, s, a)
        if Q_temp > max_Q
            a_max_Q = a
            max_Q = Q_temp
        end
    end
    return a_max_Q
end

function GetQ(Q_learning_Policy::Qlearning, s, a)
    if haskey(Q_learning_Policy._Q_table, s)
        return Q_learning_Policy._Q_table[s][a]
    else
        Q_learning_Policy._Q_table[s] = Dict{Any, Float64}()
        for a in Q_learning_Policy._action_space
            Q_learning_Policy._Q_table[s][a] = 0.0
        end
        return 0.0
    end
end

function UpdateRmaxRmin(Q_learning_Policy::Qlearning, r::Float64)
    if r > Q_learning_Policy._R_max
        Q_learning_Policy._R_max = r
    end

    if r < Q_learning_Policy._R_min
        Q_learning_Policy._R_min = r
    end
end
function EstiValueQlearning(Q_learning_Policy::Qlearning, nb_sim::Int64, s_input, pomdp, epsilon::Float64)
    a_selected = -1
    gamma = discount(pomdp)
    for i in nb_sim
        step = 0
        s = deepcopy(s_input)
        while (gamma^step) > epsilon && isterminal(pomdp, s) == false
            a_selected = ChooseActionQlearning(Q_learning_Policy, s)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a_selected)
            UpdateRmaxRmin(Q_learning_Policy, r)
            old_Q = GetQ(Q_learning_Policy, s, a_selected) 
            new_Q = old_Q + Q_learning_Policy._learning_rate * (r + gamma * MaxQ(Q_learning_Policy, sp) - old_Q)
            Q_learning_Policy._Q_table[s][a_selected] = new_Q
            s = sp
            step += 1
        end
    end

    return MaxQ(Q_learning_Policy, s_input)
end

function EstiValueQlearningGridState(Q_learning_Policy::Qlearning, nb_sim::Int64, s_input, grid_state::Vector{Float64}, pomdp, epsilon::Float64)
    a_selected = -1
    gamma = discount(pomdp)
    for i in nb_sim
        step = 0
        s = deepcopy(s_input)
        while (gamma^step) > epsilon && isterminal(pomdp, s) == false
            s_vec = convert_s(Vector{Float64}, s, pomdp)
            s_grid = ProcessStateWithGrid(grid_state, s_vec)
            a_selected = ChooseActionQlearning(Q_learning_Policy, s_grid)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a_selected)
            UpdateRmaxRmin(Q_learning_Policy, r)
            sp_vec = convert_s(Vector{Float64}, sp, pomdp)
            sp_grid = ProcessStateWithGrid(grid_state, sp_vec)
            old_Q = GetQ(Q_learning_Policy, s_grid, a_selected) 
            new_Q = old_Q + Q_learning_Policy._learning_rate * (r + gamma * MaxQ(Q_learning_Policy, sp_grid) - old_Q)
            Q_learning_Policy._Q_table[s_grid][a_selected] = new_Q
            s = sp
            step += 1
        end
    end

    s_input_grid =  ProcessStateWithGrid(grid_state, convert_s(Vector{Float64}, s_input, pomdp))
    return MaxQ(Q_learning_Policy, s_input_grid)
end


function ProcessStateWithGrid(state_grid::Vector{Float64}, state_particle::Vector{Float64})
    result = Vector{Int64}()
    for i in 1:length(state_particle)
        s_i = floor(Int64, state_particle[i] / state_grid[i])
        push!(result, s_i)
    end
    return result
end

function TrainingParallelEpisodes(
    global_policy::Qlearning,
    nb_episode_size::Int,
    nb_max_episode::Int,
    nb_samples_VMDP::Int,
    nb_sim::Int,
    epsilon::Float64,
    b0,
    pomdp;
    use_grid::Bool=false,
    grid_state=nothing
)
    improvement = typemax(Float64)
    current_avg_value = typemin(Float64)
    episode = 0

    while (improvement > epsilon) && (episode < nb_max_episode)
        println("------ Episode: ", episode , " ------")
        value_episode = 0.0

        # Create per-thread Q-learning copies
        nthreads = Threads.nthreads()
        thread_policies = [deepcopy(global_policy) for _ in 1:nthreads]
        thread_values = zeros(Float64, nthreads)

        Threads.@threads for tid in 1:nthreads
            local_policy = thread_policies[tid]
            local_value = 0.0

            # Split the work roughly equally
            for i in tid:nthreads:nb_episode_size
                for _ in 1:nb_samples_VMDP
                    s = rand(b0)
                    if use_grid
                        local_value += EstiValueQlearningGridState(local_policy, nb_sim, s, grid_state, pomdp, epsilon)
                    else
                        local_value += EstiValueQlearning(local_policy, nb_sim, s, pomdp, epsilon)
                    end
                end
            end

            thread_values[tid] = local_value
        end

        # Merge thread-local policies back into global policy
        for p in thread_policies
            merge_Q_tables!(global_policy, p)
        end

        # Compute average value for this episode
        total_value = sum(thread_values) / (nb_episode_size * nb_samples_VMDP)
        improvement = total_value - current_avg_value
        current_avg_value = total_value

        println("Avg Value: ", current_avg_value)
        episode += 1
    end
end


function merge_Q_tables!(dest::Qlearning, src::Qlearning)
    # --- Q-table ---
    for (s, qa_src) in src._Q_table
        qa_dest = get!(dest._Q_table, s, Dict{Any,Float64}())
        for (a, qv_src) in qa_src
            if haskey(qa_dest, a)
                # You can use average instead of max if preferred:
                qa_dest[a] = max(qa_dest[a], qv_src)
            else
                qa_dest[a] = qv_src
            end
        end
    end

    # --- V-table ---
    for (s, v_src) in src._V_table
        if haskey(dest._V_table, s)
            dest._V_table[s] = max(dest._V_table[s], v_src)
        else
            dest._V_table[s] = v_src
        end
    end

    # --- Learning rate & explore rate ---
    # (Simple averaging — you could also keep dest's value unchanged if you want)
    dest._learning_rate = (dest._learning_rate + src._learning_rate) / 2
    dest._explore_rate  = (dest._explore_rate  + src._explore_rate)  / 2

    # --- Action space ---
    if dest._action_space != src._action_space
        @warn "Merging Q-tables with different action spaces — keeping dest's."
    end

    # --- R_min / R_max ---
    dest._R_max = max(dest._R_max, src._R_max)
    dest._R_min = min(dest._R_min, src._R_min)

    return dest
end