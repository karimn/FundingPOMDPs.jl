doc = """
Funding POMDP simulation.

Usage:
    testpftdpw.jl <sim file> [options] [--risk-neutral | --alpha=<alpha>]

Options:
    --append, -a                               Append data
    --numprograms=<nprograms>, -p <nprograms>  Number of programs [default: 5]
    --numsim=<numsim>, -s <numsim>             Number of simulations [default: 10]
    --numsteps=<numsteps>, -t <numsteps>       Number of steps [default: 20]
    --numprocs=<nprocs>                        Number of parallel processes [default: 5]
    --depth=<depth>, -d <depth>                Planning depth [default: 10]
    --alpha=<alpha>                            Exponential utility function alpha parameter [default: 0.25]
    --risk-neutral                             Risk neutral utility
    --pftdpw-iter=<iter>                       Number of DPW solver iterations [default: 100]
    --save-pftdpw-tree                         Save MCTS tree in action info
    --k-state=<k>                              State hyperparameter k [default: 4.5]
    --reward-only                              Return rewards only in simulation data
    --use-dgp-priors                           Use the same priors for the DGP and inference.
"""

import DocOpt

args = DocOpt.docopt(
    doc, 
    isinteractive() ? "temp-data/sim_test.jls -p 2 -s 2 -t 2 --numprocs=2 --pftdpw-iter=2" : ARGS, 
    version = v"1"
)

using Distributed

# "julia -p" takes precedence
const NUM_PROCS = nprocs() > 1 ? nprocs() : parse(Int, args["--numprocs"])

if NUM_PROCS > 1 && !(nprocs() > 1)
    addprocs(NUM_PROCS, exeflags = "--project")
end

@everywhere include("FundingPOMDPs.jl")

@everywhere begin 
    using .FundingPOMDPs

    using DataFrames, StatsBase, Base.Threads, Distributions, Pipe

    import Random, Serialization
    import POMDPs, POMDPTools, POMDPSimulators
    import ParticleFilters
    import MCTS
    import Base: rand, show
    import Turing
    import ProgressMeter
end

const RNG = Random.MersenneTwister()
const NUM_PROGRAMS = parse(Int, args["--numprograms"])
const NUM_SIM_STEPS = parse(Int, args["--numsteps"])  
const NUM_TURING_MODEL_ITER = 1_000
const NUM_FILTER_PARTICLES = 2_000

dgp_priors = Priors(
    μ = Normal(0, 1.0),
    τ = Normal(0, 0.5),
    #σ = truncated(Normal(0, 0.5), 0, Inf),
    σ = InverseGamma(18.5, 30),
    η_μ = InverseGamma(26.4, 20),
    η_τ = InverseGamma(26.4, 20)
)

inference_priors = Priors(
    μ = Normal(0, 2.0),
    τ = Normal(0, 1),
    #σ = truncated(Normal(0, 1.0), 0, Inf),
    σ = truncated(Normal(0, 5.0), 0, Inf),
    η_μ = truncated(Normal(0, 2.0), 0, Inf),
    η_τ = truncated(Normal(0, 2.0), 0, Inf)
)

bayes_model = TuringModel(args["--use-dgp-priors"] ? dgp_priors : inference_priors; iter = NUM_TURING_MODEL_ITER)
bayes_updater = FundingUpdater(bayes_model)

ols_model = OlsModel()
ols_updater = FundingUpdater(ols_model)

#naive_bayes_model = TuringModel(inference_priors; iter = NUM_TURING_MODEL_ITER, multilevel = false)
#naive_bayes_updater = FullBayesianUpdater(bayes_model)

util_model = args["--risk-neutral"] ? RiskNeutralUtilityModel() : ExponentialUtilityModel(parse(Float64, args["--alpha"]))

select_subset_actionset_factory = SelectProgramSubsetActionSetFactory(NUM_PROGRAMS, 1)
explore_only_actionset_factory = ExploreOnlyActionSetFactory(NUM_PROGRAMS, 1, 1, util_model)

random_solver = POMDPTools.RandomSolver()
greedy_solver = BayesianGreedySolver()

pftdpw_solver = MCTS.DPWSolver(
    depth = parse(Int, args["--depth"]),
    exploration_constant = 25.0,
    n_iterations = parse(Int, args["--pftdpw-iter"]), #20,  #100,
    enable_action_pw = false,  
    k_state = parse(Float64, args["--k-state"]), # 4.5,
    alpha_state = 1/10.0,
    check_repeat_state = false,
    estimate_value = MCTS.RolloutEstimator(random_solver),
    keep_tree = false, # true, 
    tree_in_info = args["--save-pftdpw-tree"],
    rng = RNG 
)

const NUM_SIM = parse(Int, args["--numsim"])

planned_sims = Vector{POMDPTools.Sim}(undef, NUM_SIM)
greedy_sims = Vector{POMDPTools.Sim}(undef, NUM_SIM)
random_sims = Vector{POMDPTools.Sim}(undef, NUM_SIM)
freq_sims = Vector{POMDPTools.Sim}(undef, NUM_SIM)

pm = ProgressMeter.Progress(NUM_SIM, desc = "Preparing sims...")

@threads for sim_index in 1:NUM_SIM
    dgp_rng = Random.MersenneTwister()

    greedy_sim_rng = Random.MersenneTwister()
    planned_sim_rng = copy(greedy_sim_rng)
    random_sim_rng = copy(greedy_sim_rng)
    freq_sim_rng = copy(greedy_sim_rng)

    planned_dgp = DGP(dgp_priors, dgp_rng, NUM_PROGRAMS)
    greedy_dgp = deepcopy(planned_dgp)
    random_dgp = deepcopy(planned_dgp)
    freq_dgp = deepcopy(planned_dgp)

    init_s = Base.rand(RNG, planned_dgp; state_chain_length = NUM_SIM_STEPS + 1) # One more for the pre-state

    planned_mdp = KBanditFundingMDP{SeparateImplementEvalAction}(
        util_model,
        0.95,
        50,
        explore_only_actionset_factory,
        RNG,
        init_s
    )
    
    bayes_b = nothing

    planned_pomdp = KBanditFundingPOMDP{SeparateImplementEvalAction}(planned_mdp, bayes_model)

    particle_updater = MultiBootstrapFilter(planned_pomdp, NUM_FILTER_PARTICLES, bayes_updater)
    bayes_b = initialbelief(planned_pomdp)
    particle_b = POMDPs.initialize_belief(particle_updater, bayes_b)

    belief_mdp = MCTS.GenerativeBeliefMDP(deepcopy(planned_pomdp), particle_updater)
    pftdpw_planner = POMDPs.solve(pftdpw_solver, belief_mdp)
    random_planner = POMDPs.solve(random_solver, planned_pomdp)

    greedy_mdp = KBanditFundingMDP{ImplementOnlyAction}(
        util_model,
        0.95,
        50,
        select_subset_actionset_factory,
        RNG,
        init_s 
    )

    greedy_pomdp = KBanditFundingPOMDP{ImplementOnlyAction}(greedy_mdp, bayes_b) 
    greedy_policy = POMDPs.solve(greedy_solver, greedy_pomdp)

    freq_pomdp = KBanditFundingPOMDP{SeparateImplementEvalAction}(planned_mdp, data(bayes_b), ols_model)
    freq_random_planner = POMDPs.solve(random_solver, freq_pomdp)

    greedy_sims[sim_index] = POMDPSimulators.Sim(greedy_pomdp, greedy_policy, bayes_updater, initialbelief(greedy_pomdp), init_s, rng = greedy_sim_rng, max_steps = NUM_SIM_STEPS)
    planned_sims[sim_index] = POMDPSimulators.Sim(planned_pomdp, pftdpw_planner, particle_updater, bayes_b, init_s, rng = planned_sim_rng, max_steps = NUM_SIM_STEPS)
    random_sims[sim_index] = POMDPSimulators.Sim(planned_pomdp, random_planner, bayes_updater, bayes_b, init_s, rng = random_sim_rng, max_steps = NUM_SIM_STEPS)
    freq_sims[sim_index] = POMDPSimulators.Sim(freq_pomdp, freq_random_planner, ols_updater, initialbelief(freq_pomdp), init_s, rng = freq_sim_rng, max_steps = NUM_SIM_STEPS)

    ProgressMeter.next!(pm)
end

function create_sim_data_getter(actual_reward_only = false)
    function inner_get_sim_data(sim::POMDPTools.Sim, hist::POMDPTools.SimHistory)
        actions = collect(POMDPSimulators.action_hist(hist)) 
        beliefs = collect(POMDPSimulators.belief_hist(hist))

        sim_data = (
            state = collect(POMDPSimulators.state_hist(hist)),
            action = actions, 
            actual_reward = collect(POMDPSimulators.reward_hist(hist)),
            expected_reward = [expectedutility(rewardmodel(sim.pomdp), b, a) for (b, a) in zip(beliefs, actions)],
            total_undiscounted_actual_reward = POMDPSimulators.undiscounted_reward(hist)
        )

        if !actual_reward_only
            trees = [ s_ainfo !== nothing && haskey(s_ainfo, :tree) ? s_ainfo[:tree] : missing for s_ainfo in POMDPSimulators.ainfo_hist(hist) ]

            sim_data = (
                sim_data...,
                belief = beliefs,
                tree = trees
            )
        end

        return sim_data
    end

    return inner_get_sim_data
end

run_fun = NUM_PROCS > 1 ? POMDPTools.run_parallel : POMDPTools.run
get_sim_data = create_sim_data_getter(args["--reward-only"])

all_sim_data = @pipe vcat(greedy_sims, planned_sims, random_sims, freq_sims) |> 
    run_fun(get_sim_data, _; show_progress = true) |> 
    insertcols!(_, :plan_type => repeat(["none", "pftdpw", "random", "freq"], inner = NUM_SIM))

function append_sim_data(d, file)
    try
        d = vcat(Serialization.deserialize(file), d; cols = :union)
    catch 
        @warn "Output file doesn't exist -- creating new file" file = file 
    end

    return d
end

if args["--append"]
    global all_sim_data = append_sim_data(all_sim_data, args["<sim file>"])
end

Serialization.serialize(args["<sim file>"], all_sim_data)

