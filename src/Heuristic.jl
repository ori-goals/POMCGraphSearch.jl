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

    return current_avg_value
end



mutable struct LowerBoundPolicy{A, ASpace}
    alphas::Dict{A, Dict{Int, Float64}}
    action_space::ASpace
    R_min::Float64
end

function LowerBoundPolicy(V_table::Dict{Int, Float64}, 
                          action_space::ASpace, 
                          model::Model,  
                          max_depth::Int,
                          R_min::Float64,
                          discount::Float64;
                          nb_sim::Int64 = 10,
                          sample_states_ratio::Float64 = 0.1,
                          sample_states_threshold::Int = 100000) where {ASpace}
    A = eltype(action_space)

    num_states = length(V_table)

    if num_states > sample_states_threshold
        sampled_s = sample_keys_uniform(V_table, sample_states_ratio)
    else
        sampled_s = collect(keys(V_table))
    end

    alphas = GetBlindPolicyAlphaVectors(sampled_s, action_space, model, nb_sim, max_depth, discount)
    return LowerBoundPolicy{A, ASpace}(alphas, action_space, R_min)
end

function GetBlindPolicyAlphaVectors(all_s::Vector{Int64}, 
                                    action_space::ASpace, 
                                    model::Model,  
                                    nb_sim::Int64,
                                    max_depth::Int,
                                    discount::Float64;
                                    epsilon::Float64 = 1e-1,
                                    verbose::Bool = false) where {ASpace}
    
    A = eltype(action_space)
    all_alphas = Dict{A, Dict{Int, Float64}}()
    total_tasks = length(action_space) * length(all_s)
    completed = 0
    
    for a in action_space
        all_alphas[a] = Dict{Int, Float64}()
    end
    
    for a in action_space
        alpha_a = all_alphas[a]
        
        for s_init in all_s
            total_return = 0.0
            
            for sim_i in 1:nb_sim
                s = deepcopy(s_init)
                step = 0
                cum_return = 0.0
                
                while step < max_depth && !isterminal(model, s)
                    sp, o, r = Step(model, s, a)
                    cum_return += (discount^step) * r
                    s = sp
                    step += 1
                    
                    if (discount^step) < epsilon
                        break
                    end
                end
                
                total_return += cum_return
            end
            
            alpha_a[s_init] = total_return / nb_sim
            completed += 1
            
            if verbose && completed % max(1, total_tasks ÷ 10) == 0
                println("Progress: $(completed)/$total_tasks ($(round(completed/total_tasks*100, digits=1))%)")
            end
        end
    end
    
    return all_alphas
end

function GetValue(policy::LowerBoundPolicy, belief::OrderedDict{Int, Float64})
    max_value = -Inf
    best_action = nothing
    q_values = Dict{eltype(policy.action_space), Float64}()  
    
    for a in policy.action_space
        alpha_a = policy.alphas[a]
        value = 0.0
        
        for (s, prob) in belief
            if haskey(alpha_a, s)
                value += prob * alpha_a[s]
            else
                value += prob * policy.R_min
            end
        end
        
        q_values[a] = value  
        
        if value > max_value
            max_value = value
            best_action = a
        end
    end
    
    return max_value, best_action, q_values
end

function GetAction(policy::LowerBoundPolicy, belief::OrderedDict{Int, Float64})
    _, best_action, _ = GetValue(policy, belief)
    return best_action
end

function GetValue(policy::LowerBoundPolicy, belief::OrderedDict{Int, Float64}, action::A) where {A}
    alpha_a = policy.alphas[action]
    value = 0.0
    
    for (s, prob) in belief
        if haskey(alpha_a, s)
            value += prob * alpha_a[s]
        else
            value += prob * policy.R_min
        end        
    end
    
    return value
end

function GetAlpha(policy::LowerBoundPolicy, action::A) where {A}
    return policy.alphas[action]
end