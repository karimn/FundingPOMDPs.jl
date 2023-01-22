function get_states_data(sv::Vector{CausalState})
    @pipe DataFrame.(sv) |>
        select!.(_, :programid, :μ, :τ, :σ) |>
        vcat(_..., source = :step)
end

function get_rewards_data(sb::Vector{R}, actlist::Vector{T}, util_model::AbstractRewardModel) where {R <: Rewardable, T <: AbstractFundingAction}
    reward_data = @pipe map(enumerate(sb)) do step_state
        DataFrame(
            step = step_state[1],
            reward = expectedutility.(Ref(util_model), Ref(step_state[2]), actlist),
            actprog = [isempty(a.implement_programs) ? 0 : first(a.implement_programs) for a in actlist]
        ) 
    end |>
    vcat(_...) |>
    groupby(_, :actprog) |>
    @transform!(_, :cumul_reward = cumsum(:reward)) 

    return reward_data
end

function get_beliefs_data(bv::Vector{B}; forecast = true) where B <: AbstractBelief
    samples_fun = forecast ? state_samples : last_state_samples
    
    @pipe map(enumerate(bv)) do b
        @pipe programbeliefs(b[2]) |>
            DataFrame.(ParticleFilters.particles.(samples_fun.(_))) |>
            select!.(_, :programid, :μ, :τ, :σ) |>
            vcat(_...) |>
            @transform!(_, :step = b[1])
    end |>
        vcat(_...)
end

get_dgp_data(s::CausalState) = DataFrame(dgp(s).programdgps)