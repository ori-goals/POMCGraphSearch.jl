# This model wrapper will wrap a POMDP model and discretize the state space if it is continuous.

mutable struct Model{POMDP, S, A, O_discrete}
    pomdp::POMDP
    # for discrete state POMDPs
    Cache_s_to_index::Dict{S, Int}  # s -> sI
    Cache_index_to_s::Dict{Int, S}  # sI -> S, be careful when state is continuous
    Cache_sa_to_index::Dict{Tuple{Int,A}, Int}  # (sI,a) -> saI
    Cache_steps::Dict{Int, Dict{Tuple{Int, O_discrete}, Tuple{Float64, Float64}}}
    obs_index_cache::Dict{Int, Dict{O_discrete, Vector{Tuple{Int, Float64}}}}
    sampling_cache::Dict{Int, NamedTuple{(:outcomes, :probabilities), Tuple{Vector{Tuple{Int, O_discrete}}, Vector{Float64}}}}
    num_sim::Int
    b0_particles::Vector{Int} # initial belief particles
    # for continuous state POMDPs, we need a different structure
    bool_continuous_state::Bool
    bool_continuous_observations::Bool
    state_grid::Vector{Float64}  # discrete state grid
    obs_cluster_model::Matrix{Float64}  # observation clustering model
    Cache_svec_processed_to_index::Dict{Vector{Float64}, Int}  # s -> sI
    Cache_index_to_sraw::Dict{Int, Vector{S}}  # sI -> Vector{S}, since multiple raw states may map to the same grid state/state index
end

function isterminal(model::Model, sI::Int)
    if !model.bool_continuous_state
        s = model.Cache_index_to_s[sI] # be careful here when state is continuous
        return POMDPs.isterminal(model.pomdp, s)
    else
        # for continuous state POMDPs, randomly pick one raw state from the corresponding grid state
        s = rand(model.Cache_index_to_sraw[sI])
        return POMDPs.isterminal(model.pomdp, s)
    end
end

function discount(model::Model)
    return POMDPs.discount(model.pomdp)
end

function InitModel(pomdp::POMDP, 
                    num_sim::Int, 
                    num_b0_particles::Int; 
                    state_grid::Vector{Float64} = Vector{Float64}(),
                    O_discrete_type::Type = Int,
                    bool_continuous_observations::Bool = false,
                    obs_cluster_model::Matrix{Float64} = zeros(Float64, 0, 0)) where {POMDP} # be careful when state is continuous

    state_space_type, states = detect_state_space(pomdp)
    b0 = initialstate(pomdp)  # initial belief state
    particles = generate_initial_particles(b0, num_b0_particles)

    # for discrete state POMDPs
    Cache_s_to_index = Dict{statetype(pomdp), Int}()            # s -> sI
    Cache_index_to_s = Dict{Int, statetype(pomdp)}()            # sI -> s
    Cache_sa_to_index = Dict{Tuple{Int, actiontype(pomdp)}, Int}()  # (sI,a) -> saI
    Cache_steps = Dict{Int, Dict{Tuple{Int, O_discrete_type}, Tuple{Float64, Float64}}}() # saI -> Dict{(s', o), (pb, avg_r)}
    obs_index_cache = Dict{Int, Dict{O_discrete_type, Vector{Tuple{Int, Float64}}}}() # saI - > Dict{o, {(s', pb)}}
    sampling_cache = Dict{Int, NamedTuple{(:outcomes, :probabilities), Tuple{Vector{Tuple{Int, O_discrete_type}}, Vector{Float64}}}}()
    # for continuous state POMDPs, we need a different structure
    Cache_svec_processed_to_index = Dict{Vector{Float64}, Int}()  # s -> sI
    Cache_index_to_sraw = Dict{Int, Vector{statetype(pomdp)}}()  # sI -> Vector{S}, since multiple raw states may map to the same grid state/state index

    if state_space_type == :discrete
        # store b0 particles
        for s in particles
            if !haskey(Cache_s_to_index, s)
                sI = length(Cache_s_to_index) + 1
                Cache_s_to_index[s] = sI
                Cache_index_to_s[sI] = s
            end        
        end
        return Model(pomdp, Cache_s_to_index, Cache_index_to_s, Cache_sa_to_index, Cache_steps, obs_index_cache, sampling_cache, num_sim, [Cache_s_to_index[s] for s in particles], false, bool_continuous_observations, state_grid, obs_cluster_model, Dict{Vector{Float64}, Int}(), Dict{Int, Vector{statetype(pomdp)}}())
        # return Model(pomdp, Cache_s_to_index, Cache_index_to_s, Cache_sa_to_index, Cache_steps, num_sim, [Cache_s_to_index[s] for s in particles])
    else
        if length(state_grid) == 0
            throw(ArgumentError("For continuous state POMDPs, please provide a discrete state grid via the `state_grid` argument."))
        end
        # store b0 particles
        for s in particles
            s_vec = convert_s(Vector{Float64}, s, pomdp)
            s_vec_processed = ProcessState(s_vec, state_grid)
            if !haskey(Cache_svec_processed_to_index, s_vec_processed)
                sI = length(Cache_svec_processed_to_index) + 1
                Cache_svec_processed_to_index[s_vec_processed] = sI
                Cache_index_to_sraw[sI] = [s]
            else
                sI = Cache_svec_processed_to_index[s_vec_processed]
                push!(Cache_index_to_sraw[sI], s)
            end        
        end

        return Model(pomdp, 
        Cache_s_to_index, 
        Cache_index_to_s, 
        Cache_sa_to_index, 
        Cache_steps, 
        obs_index_cache,
        sampling_cache,
        num_sim, 
        [Cache_svec_processed_to_index[ProcessState(convert_s(Vector{Float64}, s, pomdp), state_grid)] for s in particles], 
        true, 
        bool_continuous_observations,
        state_grid, 
        obs_cluster_model,
        Cache_svec_processed_to_index, 
        Cache_index_to_sraw)
    end 

end


function Process_new_sa(model::Model{POMDP, S, A, O_discrete}, sI::Int, a::A) where {POMDP, S, A, O_discrete}
    saI = length(model.Cache_sa_to_index) + 1
    model.Cache_sa_to_index[(sI, a)] = saI
    
    model.Cache_steps[saI] = Dict{Tuple{Int, O_discrete}, Tuple{Float64, Float64}}()
    
    if !model.bool_continuous_state
        s = model.Cache_index_to_s[sI]
        for _ in 1:model.num_sim
            sp, o, r = @gen(:sp, :o, :r)(model.pomdp, s, a)
            if model.bool_continuous_observations
                o_vec = convert_o(Vector{Float64}, o, model.pomdp)
                o_processed = predict_cluster(model.obs_cluster_model, o_vec)
                o = o_processed
            end
            
            if !haskey(model.Cache_s_to_index, sp)
                spI = length(model.Cache_s_to_index) + 1
                model.Cache_s_to_index[sp] = spI
                model.Cache_index_to_s[spI] = sp
            end
            spI = model.Cache_s_to_index[sp]
            
            key = (spI, o)
            if haskey(model.Cache_steps[saI], key)
                count, total_r = model.Cache_steps[saI][key]
                model.Cache_steps[saI][key] = (count + 1.0, total_r + r)
            else
                model.Cache_steps[saI][key] = (1.0, r)
            end
        end
    else
        for _ in 1:model.num_sim
            s = rand(model.Cache_index_to_sraw[sI])
            sp, o, r = @gen(:sp, :o, :r)(model.pomdp, s, a)
            if model.bool_continuous_observations
                o_vec = convert_o(Vector{Float64}, o, model.pomdp)
                o_processed = predict_cluster(model.obs_cluster_model, o_vec)
                o = o_processed
            end
            s_vec = convert_s(Vector{Float64}, sp, model.pomdp)
            s_vec_processed = ProcessState(s_vec, model.state_grid)
            if !haskey(model.Cache_svec_processed_to_index, s_vec_processed)
                spI = length(model.Cache_svec_processed_to_index) + 1
                model.Cache_svec_processed_to_index[s_vec_processed] = spI
                model.Cache_index_to_sraw[spI] = [sp]
            else
                spI = model.Cache_svec_processed_to_index[s_vec_processed]
                push!(model.Cache_index_to_sraw[spI], sp)
            end
            
            key = (spI, o)
            if haskey(model.Cache_steps[saI], key)
                count, total_r = model.Cache_steps[saI][key]
                model.Cache_steps[saI][key] = (count + 1.0, total_r + r)
            else
                model.Cache_steps[saI][key] = (1.0, r)
            end
        end
    end
    
    total_count = model.num_sim
    for (key, (count, total_r)) in model.Cache_steps[saI]
        prob = count / total_count
        avg_r = total_r / count
        model.Cache_steps[saI][key] = (prob, avg_r)
    end
    
    build_observation_index!(model, saI)
    build_sampling_cache!(model, saI) 
end

function build_observation_index!(model::Model{POMDP, S, A, O_discrete}, saI::Int) where {POMDP, S, A, O_discrete}
    obs_index = Dict{O_discrete, Vector{Tuple{Int, Float64}}}()
    
    for ((sp, o), (prob, r)) in model.Cache_steps[saI]
        if !haskey(obs_index, o)
            obs_index[o] = [(sp, prob)]
        else
            push!(obs_index[o], (sp, prob))
        end
    end
    
    model.obs_index_cache[saI] = obs_index
end

function build_sampling_cache!(model::Model{POMDP, S, A, O_discrete}, saI::Int) where {POMDP, S, A, O_discrete}
    transitions = model.Cache_steps[saI]
    outcomes = collect(keys(transitions))
    probabilities = [transitions[outcome][1] for outcome in outcomes]
    
    model.sampling_cache[saI] = (outcomes = outcomes, probabilities = probabilities)
end


function Step(model::Model, sI::Int, a::A) where {A}
    if !haskey(model.Cache_sa_to_index, (sI, a))
        Process_new_sa(model, sI, a)
    end
    
    saI = model.Cache_sa_to_index[(sI, a)]
    
    cache = model.sampling_cache[saI]
    chosen_outcome = sample(cache.outcomes, Weights(cache.probabilities))
    
    sp, o = chosen_outcome
    r = model.Cache_steps[saI][chosen_outcome][2] 
    
    return (sp, o, r)
end


function Step_batch(model::Model, sI::Int, a::A) where {A}
    if !haskey(model.Cache_sa_to_index, (sI, a))
        Process_new_sa(model, sI, a)
    end

    return model.Cache_steps[model.Cache_sa_to_index[(sI, a)]]
end


