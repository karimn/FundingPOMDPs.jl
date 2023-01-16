begin
    include("FundingPOMDPs.jl")

    using .FundingPOMDPs
    using DataFrames, DataFramesMeta
    using StatsBase, Gadfly, Pipe, Serialization
    using ParticleFilters
    using SplitApplyCombine

    import StatsPlots
end

file_suffix = ""

greedy_sim_data = deserialize("src/greedy_sim$(file_suffix).jls")
pftdpw_sim_data = deserialize("src/pftdpw_sim$(file_suffix).jls")

util_diff = map((g, p) -> p - g, greedy_sim_data.actual_reward, pftdpw_sim_data.actual_reward)
util_diff_mean = [mean(a) for a in invert(util_diff)]
util_diff_quant = @pipe [quantile(a, [0.25, 0.5, 0.75]) for a in invert(util_diff)] |>
    DataFrame(invert(_), [:lb, :med, :ub]) |>
    insertcols!(_, :step => 1:nrow(_))

# Plot sim util diff and median util diff
@pipe [DataFrame(sim = i, step = 1:length(util_diff[i]), diff = util_diff[i]) for i in 1:length(util_diff)] |>
    vcat(_...) |>
    #@subset(_, :sim .== 13) |>
    plot(
        layer(util_diff_quant, x = :step, y = :med, ymin = :lb, ymax = :ub,  color = [colorant"red"], alpha = [0.75], Geom.line, Geom.ribbon),
        layer(y = util_diff_mean, color = [colorant"darkgreen"], Geom.point, Geom.line),
        layer(_, x = :step, y = :diff, group = :sim, alpha = [0.25], Geom.line),
        layer(yintercept = [0.0], Geom.hline(style = :dot, color = "grey")),
        Scale.x_discrete
    )

# Plot actual rewards
@pipe 25 |> 
    pairs((greedy = greedy_sim_data.actual_reward[_], planned = pftdpw_sim_data.actual_reward[_])) |>
    DataFrame(_) |>
    @transform!(_, :step = 1:nrow(_)) |>
    stack(_, [:greedy, :planned]) |>
    groupby(_, :variable) |>
    @transform!(_, :cumul_reward = cumsum(:value)) |>
    #plot(_, x = :step, y = :value, color = :variable, Geom.point, Geom.line, Scale.x_discrete)
    plot(_, x = :step, y = :cumul_reward, color = :variable, Geom.point, Geom.line, Scale.x_discrete)

function get_state_and_belief_data(state_arr)
    states = @pipe DataFrame.(state_arr) |> 
        @.select(_, :programid, :μ, :τ, :σ)
    for i in 1:length(states)
        states[i].step .= i
    end

    return vcat(states...)
end

x_greedy = @pipe get_state_and_belief_data.(greedy_sim_data.state) |> 
    [@transform!(y, :μ_control = :μ, :μ_treated = :μ + :τ) for y in _]

x_plan = @pipe get_state_and_belief_data.(pftdpw_sim_data.state) |> 
    [@transform!(y, :μ_control = :μ, :μ_treated = :μ + :τ) for y in _]
    
@pipe x_greedy[1] |>
    stack(_, [:μ_control, :μ_treated]; variable_name = :arm) |>
    select!(_, :programid, r"μ_", :step, :arm, :value) |>
    transform!(_, :arm => (arm -> [m.captures[1] for m in match.(r"_(control|treated)", arm)]) => :arm) |>
    plot(
        _, x = :step, y = :value, group = :programid, color = :programid, 
        ygroup = :arm,
        Geom.subplot_grid(Geom.line),
        Scale.color_discrete, Scale.group_discrete
        #layer(_, x = :step, y = :μ_treated , color = :red, Geom.line)
    )

@pipe x_greedy[17] |>
    @rsubset(_, :programid in [2, 9, 4]) |>
    plot(
        _, x = :step, 
        layer(y = :τ, Geom.line), 
#        layer(y = :μ, Geom.line), 
        color = :programid, Scale.color_discrete, Scale.x_discrete, Guide.title("Greedy")
    )
    
@pipe x_plan[26] |>
    @rsubset(_, :programid in [5, 6, 10]) |>
    plot(
        _, x = :step, 
        layer(y = :τ, Geom.line), 
#        layer(y = :μ, Geom.line), 
        color = :programid, Scale.color_discrete, Scale.x_discrete, Guide.title("Planned")
    )

greedy_sim_data.action[3][12]
pftdpw_sim_data.action[3][12]

greedy_sim_data.actual_reward[11][10]
pftdpw_sim_data.actual_reward[11][10]

# Beliefs #####################

b_data_greedy = @pipe map(enumerate(greedy_sim_data.belief[11])) do sb 
    @pipe sb[2].progbeliefs |>
        [DataFrame(ParticleFilters.particles(spb.state_samples)) for spb in _] |>
        select!.(_, :programid, :μ, :τ, :σ) |>
        vcat(_...) |>
        @transform(_, :step = sb[1])
end |>
    vcat(_...)
    
b_data_planned = @pipe map(enumerate(pftdpw_sim_data.belief[26])) do sb 
    @pipe sb[2].progbeliefs |>
        [DataFrame(ParticleFilters.particles(spb.state_samples)) for spb in _] |>
        select!.(_, :programid, :μ, :τ, :σ) |>
        vcat(_...) |>
        @transform(_, :step = sb[1])
end |>
    vcat(_...)

@pipe b_data_greedy |>
    @subset(_, :step .== 12) |>
    vstack(
        plot(_, x = :programid, y = :τ, Geom.boxplot, Scale.x_discrete),
        plot(_, x = :programid, y = :μ, Geom.boxplot, Scale.x_discrete),
    )

pftdpw_sim_data.belief[3][12].progbeliefs[3].data
pftdpw_sim_data.state[3][1].programstates
ds = getdatasets(Base.rand(Random.GLOBAL_RNG, pftdpw_sim_data.state[3][1], 50))[3]

@pipe b_data_greedy |>
    @rsubset(_, :step == 12, :programid in (3, 10)) |>
    plot(_, x = :τ, color = :programid, alpha = [0.75], Geom.histogram(position = :identity, density = true), Scale.discrete_color)

@pipe b_data_planned |>
    @rsubset(_, :step == 10, :programid in (1, 2)) |>
    plot(_, x = :τ, color = :programid, alpha = [0.75], Geom.histogram(position = :identity), Scale.discrete_color)

@pipe b_data_planned |>
    @rsubset(_, :step == 10, :programid in (1, 2)) |>
    plot(_, x = :σ, color = :programid, alpha = [0.75], Geom.histogram(position = :identity), Scale.discrete_color)

@pipe b_data_planned |>
    @subset(_, :programid .== 1) |>
    @rsubset(_, :step in (1, 10)) |>
    plot(_, x = :τ, color = :step, alpha = [0.75], Geom.histogram(density = true, position = :identity), Scale.color_discrete)

@pipe b_data_planned |>
    @subset(_, :programid .== 5) |>
    @rsubset(_, :step in (1, 3)) |>
    plot(_, x = :τ, color = :step, alpha = [0.5], Geom.histogram(density = true, position = :identity), Scale.color_discrete)

@pipe pairs((greedy = b_data_greedy, planned = b_data_planned)) |>
    [@transform(v, :src = k) for (k, v) in _] |>
    vcat(_...) |>
    @subset(_, :programid .== 4, :step .== 1) |>
    plot(
        _,
        x = :τ, color = :src, alpha = [0.75],
        Geom.histogram(density = true, position = :identity)
    )


util_model = ExponentialUtilityModel(1.0)

@pipe greedy_sim_data.belief[11][1] |>
    _.progbeliefs |>
    expectedutility.(Ref(util_model), _, false)

@pipe greedy_sim_data.state[1][1] |>
    _.programstates |>
    expectedutility.(Ref(util_model), _, true)

b = greedy_sim_data.belief[11][10]
a = greedy_sim_data.action[1][1]
pb2 = b.progbeliefs[2]
@pipe pb2 |> expectedutility.(Ref(util_model), Ref(_), [false, true])
pb2p = @pipe ParticleFilters.particles(pb2.state_samples) |>
    @transform!(
        DataFrame(_), 
        eu0 = expectedutility.(Ref(util_model), _, false), 
        eu1 = expectedutility.(Ref(util_model), _, true)
    ) |>
    select!(_, :μ, :τ, :σ, r"eu") |>
    @transform!(_, eu_diff = :eu1 - :eu0)

@pipe pb2p |>
    plot(_, x = :eu_diff, Geom.histogram)

expectedutility(util_model, b, a)
expectedutility(util_model, b, ImplementOnlyAction())


asf = SelectProgramSubsetActionSetFactory(10, 1)
actlist = actions(asf).actions

function get_rewards_data(sb)
    reward_data = @pipe map(enumerate(sb)) do sim_states
        @pipe map(enumerate(sim_states[2])) do step_state
            @pipe DataFrame(
                reward = expectedutility.(Ref(util_model), Ref(step_state[2]), actlist),
                actprog = [isempty(a.implement_programs) ? 0 : first(a.implement_programs) for a in actlist] 
            ) |>
                @transform!(_, :step = step_state[1])
        end |>
        vcat(_...) |>
        @transform!(_, :sim = sim_states[1])
    end |>
    vcat(_...) |>
    groupby(_, Cols(:sim, :actprog)) |>
    @transform!(_, :cumul_reward = cumsum(:reward)) 

    return reward_data
end

all_rewards = get_rewards_data(greedy_sim_data.state)
all_expected_rewards = get_rewards_data(pftdpw_sim_data.belief) 

@pipe all_rewards |>
    @subset(_, :sim .== 26) |>
    plot(
        _, x = :step, y = :reward, color = :actprog,
        Geom.line, Scale.color_discrete, Scale.x_discrete,
        Guide.title("True Rewards")
    )

@pipe all_expected_rewards |>
    @subset(_, :sim .== 26) |>
    plot(
        _, x = :step, y = :reward, color = :actprog,
        Geom.line, Scale.color_discrete, Scale.x_discrete,
        Guide.title("Expected Rewards")
    )

@pipe all_rewards |>
    @subset(_, :sim .== 21) |>
    plot(
        _, x = :step, y = :cumul_reward, color = :actprog,
        Geom.line, Scale.color_discrete
    )


@pipe pftdpw_sim_data.belief[26][1].progbeliefs[5] |>
    DataFrame(u0 = utility_particles(util_model, _, false), u1 = utility_particles(util_model, _, true)) |>
    stack(_) |>
    #filter.(u -> u > -10_000, _) |>
    plot(_, x = :value, color = :variable, alpha = [0.5], Geom.histogram(position = :identity))

@pipe pftdpw_sim_data.belief[26][1].progbeliefs[5].data[1].y_treated |>
    utility.(Ref(util_model), _) |>
    filter(u -> u > -10_000, _) |>
    plot(x = _, Geom.histogram)

@pipe pftdpw_sim_data.belief[26][1].progbeliefs[5]