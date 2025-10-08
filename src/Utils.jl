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



function NormalizeDict(d::OrderedDict{Any, Float64})
	sum = 0.0
	for (key, value) in d
		sum += value
	end

	for (key, value) in d
		d[key] = value / sum
	end
end

function NormalizeDict(d::Dict{Int64, Float64})
	sum = 0.0
	for (key, value) in d
		sum += value
	end

	for (key, value) in d
		d[key] = value / sum
	end
end


function FindRLower(pomdp, b0, action_space; nb_sim::Int64=100, epsilon::Float64=0.01)
	action_min_r = Dict{Any, Float64}()
	for a in action_space
		min_r = typemax(Float64)
		for i in 1:nb_sim
			s = rand(b0)
			step = 0
			while (discount(pomdp)^step) > epsilon && isterminal(pomdp, s) == false
				sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a)
				s = sp
				if r < min_r
					action_min_r[a] = r
					min_r = r
				end
				step += 1
			end
		end
	end

	max_min_r = typemin(Float64)
	for a in action_space
		if (action_min_r[a] > max_min_r)
			max_min_r = action_min_r[a]
		end
	end

	return max_min_r / (1 - discount(pomdp))
end


function ProcessState(s_vec::Vector{Float64}, state_grid::Vector{Float64})
    result = Vector{Int64}()
    for i in 1:length(s_vec)
        s_i = floor(Int64, s_vec[i] / state_grid[i])
        push!(result, s_i)
    end
    return result
end
