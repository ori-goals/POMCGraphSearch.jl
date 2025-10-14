mutable struct Qlearning{A}
    _Q_table::Dict{Int, Dict{A, Float64}} #s -> a -> Q
    _V_table::Dict{Int, Float64} # s -> V
    _learning_rate::Float64
    _explore_rate::Float64
    _action_space
    _R_max::Float64
    _R_min::Float64
end

function ChooseActionQlearning(Q_learning_Policy::Qlearning, s::Int)
    rand_num = rand()
    if rand_num < Q_learning_Policy._explore_rate
        a_selected = rand(Q_learning_Policy._action_space)
    else
        a_selected = BestAction(Q_learning_Policy::Qlearning, s)
    end
    return a_selected
end

function MaxQ(Q_learning_Policy::Qlearning, s::Int)
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

function GetV(Q_learning_Policy::Qlearning, s::Int)
    if haskey(Q_learning_Policy._V_table, s)
        return Q_learning_Policy._V_table[s]
    else
        return 0.0
    end
end

function BestAction(Q_learning_Policy::Qlearning, s::Int) 
    max_Q = typemin(Float64)
    a_max_Q = nothing
    for a in Q_learning_Policy._action_space
        Q_temp = GetQ(Q_learning_Policy, s, a)
        if Q_temp > max_Q
            a_max_Q = a
            max_Q = Q_temp
        end
    end
    return a_max_Q
end

function GetQ(Q_learning_Policy::Qlearning, s::Int, a::A) where {A}
    if haskey(Q_learning_Policy._Q_table, s)
        return Q_learning_Policy._Q_table[s][a]
    else
        Q_learning_Policy._Q_table[s] = Dict{A, Float64}()
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


function EstiValueQlearning(Q_learning_Policy::Qlearning, nb_sim::Int64, s_input::Int, model::Model, epsilon::Float64)
    a_selected = -1
    gamma = discount(model)
    for i in nb_sim
        step = 0
        s = deepcopy(s_input)
        while (gamma^step) > epsilon && isterminal(model, s) == false
            a_selected = ChooseActionQlearning(Q_learning_Policy, s)
            sp, o, r = Step(model, s, a_selected)
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

function ProcessStateWithGrid(state_grid::Vector{Float64}, state_particle::Vector{Float64})
    result = Vector{Int64}()
    for i in 1:length(state_particle)
        s_i = floor(Int64, state_particle[i] / state_grid[i])
        push!(result, s_i)
    end
    return result
end

function TrainingEpisodes(
    global_policy::Qlearning,
    nb_episode_size::Int,
    nb_max_episode::Int,
    nb_samples_VMDP::Int,
    nb_sim::Int,
    epsilon::Float64,
    model::Model
    )

    improvement = typemax(Float64)
    current_avg_value = typemin(Float64)
    episode = 0

    b0 = model.b0_particles

    while (improvement > epsilon) && (episode < nb_max_episode)
        println("------ Episode: ", episode, " ------")

        value_episode = 0.0

        # Update the same policy directly
        for i in 1:nb_episode_size
            for _ in 1:nb_samples_VMDP
                s = rand(b0)
                value_episode += EstiValueQlearning(global_policy, nb_sim, s, model, epsilon)
            end
        end

        total_value = value_episode / (nb_episode_size * nb_samples_VMDP)
        improvement = total_value - current_avg_value
        current_avg_value = total_value

        println("Avg Value: ", current_avg_value)
        episode += 1
    end
end
