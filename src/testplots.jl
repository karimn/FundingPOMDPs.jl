begin
    include("FundingPOMDPs.jl")

    using .FundingPOMDPs
    using DataFrames, DataFramesMeta
    using StatsBase, Gadfly, Pipe, Serialization
    using ParticleFilters
    using SplitApplyCombine
    using Distributions

    import StatsPlots
end

Gadfly.set_default_plot_size(60cm, 40cm)

file_suffix = ""

util_model = ExponentialUtilityModel(0.5)
#util_model = RiskNeutralUtilityModel()

greedy_sim_data = deserialize("src/greedy_sim$(file_suffix).jls")
pftdpw_sim_data = deserialize("src/pftdpw_sim$(file_suffix).jls")

util_diff = map((g, p) -> p - g, greedy_sim_data.actual_reward, pftdpw_sim_data.actual_reward)
util_diff_mean = [mean(a) for a in invert(util_diff)]
util_diff_quant = @pipe [quantile(a, [0.25, 0.5, 0.75]) for a in invert(util_diff)] |>
    DataFrame(invert(_), [:lb, :med, :ub]) |>
    insertcols!(_, :step => 1:nrow(_))

obs_reward = @pipe pairs((greedy = greedy_sim_data.actual_reward, planned = pftdpw_sim_data.actual_reward)) |>
    DataFrame(_) |>
    insertcols!(_, :sim => 1:nrow(_)) |>
    DataFrames.flatten(_, [:greedy, :planned]) |>
    groupby(_, :sim) |>
    transform!(_, eachindex => :step) |>
    stack(_, [:greedy, :planned], value_name = :reward) |>
    groupby(_, Cols(:sim, :variable)) |>
    @transform!(_, :cumul_reward = cumsum(:reward))

nsteps = maximum(obs_reward.step)

b_data_greedy = @pipe get_beliefs_data.(greedy_sim_data.belief) |> 
    vcat(_..., source = :sim)

nprograms = maximum(b_data_greedy.programid)

b_data_planned = @pipe get_beliefs_data.(pftdpw_sim_data.belief) |> 
    vcat(_..., source = :sim)

last_b_data_greedy = @pipe get_beliefs_data.(greedy_sim_data.belief; forecast = false) |> 
    vcat(_..., source = :sim)

last_b_data_planned = @pipe get_beliefs_data.(pftdpw_sim_data.belief; forecast = false) |> 
    vcat(_..., source = :sim)

summarize_b_data(b_data, probs = [0.1, 0.5, 0.9]) = @pipe b_data |>
    groupby(_, Cols(:sim, :programid, :step)) |>
    @combine(_, :per = probs, :quant_τ = quantile(:τ, probs)) |>
    unstack(_, :per, :quant_τ; renamecols = c -> "per_$c") 

b_data_greedy_summ = summarize_b_data(b_data_greedy) 
b_data_planned_summ = summarize_b_data(b_data_planned)  
last_b_data_greedy_summ = summarize_b_data(last_b_data_greedy) 
last_b_data_planned_summ = summarize_b_data(last_b_data_planned)  

s_data = @pipe get_states_data.(greedy_sim_data.state) |>
    vcat(_..., source = :sim)

# Plot sim util diff and median util diff
@pipe [DataFrame(sim = i, step = 1:length(util_diff[i]), diff = util_diff[i]) for i in 1:length(util_diff)] |>
    vcat(_...) |>
    #@subset(_, :sim .== 13) |>
    plot(
        layer(util_diff_quant, x = :step, y = :med, ymin = :lb, ymax = :ub,  color = [colorant"red"], alpha = [0.75], Geom.line, Geom.ribbon),
        layer(y = util_diff_mean, color = [colorant"darkgreen"], Geom.point, Geom.line),
        layer(_, x = :step, y = :diff, group = :sim, alpha = [0.25], Geom.line),
        layer(yintercept = [0.0], Geom.hline(style = :dot, color = "grey")),
        Scale.x_discrete, Guide.yticks(ticks = -5:0.5:5) 
    )

actlist = @pipe SelectProgramSubsetActionSetFactory(nprograms, 1) |> actions(_).actions

all_rewards = @pipe get_rewards_data.(greedy_sim_data.state, Ref(actlist), Ref(util_model)) |>
    [@transform!(rd[2], :sim = rd[1]) for rd in enumerate(_)] |>
    vcat(_...) |>
    insertcols!(_, :reward_type => "actual")

greedy_all_expected_rewards = @pipe get_rewards_data.(greedy_sim_data.belief, Ref(actlist), Ref(util_model)) |>
    [@transform!(rd[2], :sim = rd[1]) for rd in enumerate(_)] |>
    vcat(_...) |>
    insertcols!(_, :reward_type => "expected greedy")

all_expected_rewards = @pipe get_rewards_data.(pftdpw_sim_data.belief, Ref(actlist), Ref(util_model)) |>
    [@transform!(rd[2], :sim = rd[1]) for rd in enumerate(_)] |>
    vcat(_...) |>
    insertcols!(_, :reward_type => "expected planned")

#obs_act = @pipe pairs((:planned => pftdpw_sim_data.action, :greedy => greedy_sim_data.action)) |>
obs_act = @pipe pftdpw_sim_data.action |>
    DataFrame.(_) |>
    [@transform!(rd[2], :sim = rd[1]) for rd in enumerate(_)] |>
    vcat(_...) |>
    @rtransform(
        _, 
        :implement_programs = isempty(:implement_programs) ? 0 : first(:implement_programs),
        :eval_programs = isempty(:eval_programs) ? 0 : first(:eval_programs)
    ) |>
    groupby(_, :sim) |>
    transform!(_, eachindex => :step) |>
    stack(_, [:implement_programs, :eval_programs], variable_name = :action_type, value_name = :pid)

pdgps_data = @pipe pftdpw_sim_data.state |>
    get_dgp_data.(_[1]) |>
    [ @transform!(d[2], :sim = d[1]) for d in enumerate(_)] |>
    vcat(_...)

begin
    sim = 8 
    
    obs_reward_plot = @pipe obs_reward |>
        @rsubset(_, :sim in sim) |>
        #plot(_, x = :step, y = :value, color = :variable, Geom.point, Geom.line, Scale.x_discrete)
        plot(_, x = :step, y = :reward, xgroup = :sim, color = :variable, Geom.subplot_grid(Geom.point, Geom.line), Scale.x_discrete)

    obs_cumul_reward_plot = @pipe obs_reward |>
        @rsubset(_, :sim in sim) |>
        #plot(_, x = :step, y = :value, color = :variable, Geom.point, Geom.line, Scale.x_discrete)
        plot(_, x = :step, y = :cumul_reward, xgroup = :sim, color = :variable, Geom.subplot_grid(Geom.point, Geom.line), Scale.x_discrete)

    reward_plot = @pipe vcat(all_rewards, all_expected_rewards, greedy_all_expected_rewards) |>
        @subset(_, :sim .== sim, :reward_type .== "actual") |>
        plot(
            _, x = :step, y = :reward, color = :actprog, linestyle = :reward_type, 
            Geom.line, Scale.color_discrete, Scale.x_discrete,
            Guide.title("True Rewards"), 
        )

    cumul_reward_plot =  
        @pipe vcat(all_rewards, all_expected_rewards, greedy_all_expected_rewards) |>
            @subset(_, :sim .== sim, :reward_type .== "actual") |>
            plot(
                _, x = :step, y = :cumul_reward, color = :actprog, linestyle = :reward_type,
                Geom.line, Scale.color_discrete, Scale.x_discrete,
                Guide.title("Cumulative Reward")
            )

    act_plot = @pipe obs_act |>
        @rsubset(_, :sim in sim) |>
        @rtransform!(_, :pid = :action_type == "implement_programs" ? :pid - 0.02 : :pid + 0.02) |>
        plot(
            _, x = :step, y = :pid, color = :action_type, 
            Geom.step, style(line_width = 1mm), Scale.x_discrete, Coord.cartesian(ymin = 0, ymax = nprograms)
        )

    belief_plot = @pipe obs_act |>
        @subset(_, :sim .== sim, :action_type .== "eval_programs") |>
        @transform!(_, :step = :step .+ 1) |>
        vcat(
            @subset(b_data_planned_summ, :sim .== sim, :step .== 1),
            rightjoin(b_data_planned_summ, _, on = [:sim, :step, (:programid => :pid)]);
            cols = :union
        ) |>
        plot(
            _, x = :step, ymin = "per_0.1", ymax = "per_0.9", color = :programid,
            layer(alpha = [0.2], Geom.ribbon, Stat.dodge),
            layer(Geom.yerrorbar, Stat.dodge, style(line_width = 1mm, errorbar_cap_length=0mm)), 
            layer(@subset(s_data, :sim .== sim), y = :τ, Geom.line, Stat.dodge),
            Scale.discrete_color, Scale.x_discrete(levels = 1:(nsteps + 1)), Guide.ylabel("τ"),
            Guide.title("Predicted States")
        )
        
    last_belief_plot = @pipe obs_act |>
        @subset(_, :sim .== sim, :action_type .== "eval_programs") |>
        @transform!(_, :step = :step .+ 1) |>
        vcat(
            @subset(last_b_data_planned_summ, :sim .== sim, :step .== 1),
            rightjoin(last_b_data_planned_summ, _, on = [:sim, :step, (:programid => :pid)]);
            cols = :union
        ) |>
        @transform!(_, :step = :step .- 1) |>
        plot(
            _, x = :step, y = "per_0.5", ymin = "per_0.1", ymax = "per_0.9", color = :programid,
            layer(Geom.line),
            layer(alpha = [0.8], Geom.ribbon),
            layer(Geom.yerrorbar, style(line_width = 1mm, errorbar_cap_length=0mm)), 
            Scale.discrete_color, Scale.x_discrete(levels = 0:nsteps), Guide.ylabel("τ"),
            Guide.title("Last States")
        )

    τ_density_data = @pipe pdgps_data |>
        @subset(_, :sim .== sim) |>
        @rselect!(_, :programid, :τ = rand(Distributions.Normal(:τ, :η_τ), 10000)) |>
        DataFrames.flatten(_, :τ) 

    first_belief_hist = @pipe pairs((planned = b_data_planned, greedy = b_data_greedy)) |>
        [@transform(v, :policy = k) for (k, v) in _] |>
        vcat(_...) |>
        @subset(_, :sim .== sim, :step .== 1) |>
        plot(_, x = :τ, xgroup = :programid, alpha = [0.5], color = :policy, Geom.subplot_grid(Geom.histogram(position = :identity, density = true), Scale.x_continuous(minvalue = -5, maxvalue = 5)))

    first_belief = @pipe b_data_planned_summ |>
        @subset(_, :sim .== sim, :step .== 1) |>
        #plot(_, x = :τ, xgroup = :programid, alpha = [0.5], Geom.subplot_grid(Geom.histogram(density = true), Scale.x_continuous(minvalue = -5, maxvalue = 5)))
        plot(
            layer(_, x = :programid, y = "per_0.5", ymin = "per_0.1", ymax = "per_0.9", Geom.errorbar, Geom.point, style(errorbar_cap_length=0mm, line_width = 0.5mm)),
            layer(τ_density_data, x = :programid, y = :τ, alpha = [0.1], Geom.violin),
            Scale.x_discrete
        )

    title(gridstack([obs_reward_plot   obs_cumul_reward_plot; 
                     reward_plot       cumul_reward_plot; 
                     act_plot          belief_plot; 
                     first_belief_hist last_belief_plot]), "Simulation $sim")
end

#@pipe map(0:0.1:2) do α
@pipe map([0.5]) do α
    um = ExponentialUtilityModel(α)

    @pipe DataFrame(α = α, x = -1.5:0.1:1.5) |>
        @transform!(_, :u = utility.(Ref(um), :x))
end |>
    vcat(_...) |>
    plot(_, x = :x , y = :u, color = :α, Geom.line, Coord.cartesian(fixed = true))