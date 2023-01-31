doc = """
Funding POMDP simulation.

Usage:
    testpftdpw.jl <greedy file> <planned file> [options] [--risk-neutral | --alpha=<alpha>]

Options:
    --append, -a                               Append data
    --numprograms=<nprograms>, -p <nprograms>  Number of programs [default: 5]
    --numsim=<numsim>, -s <numsim>             Number of simulations [default: 10]
    --numsteps=<numsteps>, -t <numsteps>       Number of steps [default: 20]
    --numprocs=<nprocs>                        Number of parallel processes [default: 5]
    --depth=<depth>, -d <depth>                Planning depth [default: 10]
    --alpha=<alpha>                            Exponential utility function alpha parameter [default: 0.25]
    --risk-neutral                             Risk neutral utility
    --plan-algo=<algo>                         Planning algorithm (pftdpw, random) [default: pftdpw]
    --pftdpw-iter=<iter>                       Number of DPW solver iterations [default: 50]
    --save-pftdpw-tree                         Save MCTS tree in action info
"""

import DocOpt

args = DocOpt.docopt(
    doc, 
    isinteractive() ? "greedy_sim_test.jls pftdpw_sim_test.jls -p 5 -s 2 -t 3 --numprocs=1 --plan-algo=random" : ARGS, 
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

    using DataFrames, StatsBase, Base.Threads, Distributions

    import Random, Serialization
    import POMDPs, POMDPTools, POMDPSimulators
    import ParticleFilters
    import MCTS
    import Base: rand, show
    import Turing
end

const RNG = Random.MersenneTwister()
const NUM_PROGRAMS = parse(Int, args["--numprograms"])
const NUM_SIM = parse(Int, args["--numsim"])
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
    k_state = 4.5,
    alpha_state = 1/10.0,
    check_repeat_state = false,
    estimate_value = MCTS.RolloutEstimator(random_solver),
    keep_tree = false, # true, 
    tree_in_info = args["--save-pftdpw-tree"],
    rng = RNG 
)

bayes_model = TuringModel(inference_priors; iter = NUM_TURING_MODEL_ITER)
bayes_updater = FullBayesianUpdater(RNG, bayes_model)

#naive_bayes_model = TuringModel(inference_hyperparam; multilevel = false)
#naive_bayes_updater = FullBayesianUpdater(RNG, bayes_model)

planned_sims = Vector{POMDPTools.Sim}(undef, NUM_SIM)
greedy_sims = Vector{POMDPTools.Sim}(undef, NUM_SIM)

@threads for sim_index in 1:NUM_SIM
    dgp_rng = Random.MersenneTwister()

    planned_dgp = DGP(dgp_priors, dgp_rng, NUM_PROGRAMS)
    greedy_dgp = deepcopy(planned_dgp)

    init_s = Base.rand(RNG, planned_dgp; state_chain_length = NUM_SIM_STEPS)

    planned_mdp = KBanditFundingMDP{SeparateImplementEvalAction}(
        util_model,
        0.95,
        50,
        planned_dgp,
        explore_only_actionset_factory;
        curr_state = init_s
    )

    planned_pomdp = KBanditFundingPOMDP{SeparateImplementEvalAction, FullBayesianBelief}(planned_mdp, bayes_model)

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
        greedy_dgp,
        select_subset_actionset_factory;
        curr_state = init_s 
    )

    greedy_pomdp = KBanditFundingPOMDP{ImplementOnlyAction, FullBayesianBelief}(greedy_mdp, bayes_b) #, bayes_model)

    greedy_policy = POMDPs.solve(greedy_solver, greedy_pomdp)

    greedy_sim_rng = Random.MersenneTwister()
    planned_sim_rng = copy(greedy_sim_rng)

    if args["--plan-algo"] == "pftdpw"
        planned_sims[sim_index] = POMDPSimulators.Sim(planned_pomdp, pftdpw_planner, particle_updater, bayes_b, init_s, rng = planned_sim_rng, max_steps = NUM_SIM_STEPS)
    elseif args["--plan-algo"] == "random"
        planned_sims[sim_index] = POMDPSimulators.Sim(planned_pomdp, random_planner, bayes_updater, bayes_b, init_s, rng = planned_sim_rng, max_steps = NUM_SIM_STEPS)
    else
        error("Unknown planning algorithm '$(args["--plan-algo"])'")
    end

    greedy_sims[sim_index] = POMDPSimulators.Sim(greedy_pomdp, greedy_policy, bayes_updater, initialbelief(greedy_pomdp), init_s, rng = greedy_sim_rng, max_steps = NUM_SIM_STEPS)
end

@everywhere function get_sim_data(sim::POMDPTools.Sim, hist::POMDPTools.SimHistory)
    actions = collect(POMDPSimulators.action_hist(hist)) 
    beliefs = collect(POMDPSimulators.belief_hist(hist))

    return (
        action = actions, 
        actual_reward = collect(POMDPSimulators.reward_hist(hist)),
        belief = beliefs,
        expected_reward = [expectedutility(rewardmodel(sim.pomdp), b, a) for (b, a) in zip(beliefs, actions)],
        state = collect(POMDPSimulators.state_hist(hist)),

        total_undiscounted_actual_reward = POMDPSimulators.undiscounted_reward(hist)
    )
end

run_fun = NUM_PROCS > 1 ? POMDPTools.run_parallel : POMDPTools.run

greedy_sim_data = run_fun(get_sim_data, greedy_sims; show_progress = true)
planned_sim_data = run_fun(get_sim_data, planned_sims; show_progress = true) 

if args["--append"]
    try
        global greedy_sim_data = vcat(Serialization.deserialize(args["<greedy file>"]), greedy_sim_data)
    catch 
        @warn "Output file doesn't exist -- creating new file" file = args["<greedy file>"]
    end

    try
        global planned_sim_data = vcat(Serialization.deserialize(args["<planned file>"]), planned_sim_data)
    catch 
        @warn "Output file doesn't exist -- creating new file" file = args["<planned file>"]
    end

end

Serialization.serialize(args["<greedy file>"], greedy_sim_data)
Serialization.serialize(args["<planned file>"], planned_sim_data)

#x = Serialization.deserialize("greedy_sim_test.jls")
#pftdpw_sim_data = Serialization.deserialize("pftdpw_sim.jls")

#Serialization.deserialize("Code/funding-portfolio/src/greedy_sim.jls")
