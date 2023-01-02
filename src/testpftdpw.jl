using Distributed

addprocs(5, exeflags = "--project")

@everywhere include("FundingPOMDPs.jl")

@everywhere begin 
    using .FundingPOMDPs

    using DataFrames, StatsBase, Base.Threads

    import Random
    import POMDPs, POMDPTools, POMDPSimulators
    import ParticleFilters
    import MCTS
    import Base: rand, show
    import Turing
end

const RNG = Random.MersenneTwister()
const NUM_PROGRAMS = 5 
const NUM_SIM = 10 
const NUM_SIM_STEPS = 10 
const NUM_FILTER_PARTICLES = 2_000

dgp_hyperparam = Hyperparam(mu_sd = 1.0, tau_mean = 0.1, tau_sd = 0.25, sigma_sd = 1.0, eta_sd = [0.1, 0.1, 0.1])
inference_hyperparam = Hyperparam(mu_sd = 2.0, tau_mean = 0.0, tau_sd = 0.5, sigma_sd = 4.0, eta_sd = [0.2, 0.2, 0.2])

util_model = ExponentialUtilityModel(1.0)

select_subset_actionset_factory = SelectProgramSubsetActionSetFactory(NUM_PROGRAMS, 1)
explore_only_actionset_factory = ExploreOnlyActionSetFactory(NUM_PROGRAMS, 1, 1, util_model)

dpfdpw_solver = MCTS.DPWSolver(
    depth = 10,
    exploration_constant = 0.0,
    n_iterations = 100,
    enable_action_pw = false,  
    k_state = 4.5,
    alpha_state = 1/10.0,
    check_repeat_state = false,
    estimate_value = MCTS.RolloutEstimator(MCTS.RandomSolver(RNG)),
    keep_tree = true,
    rng = RNG 
)

greedy_solver = BayesianGreedySolver()

bayes_updater = FullBayesianUpdater(inference_hyperparam, RNG)

pftdpw_sims = Vector{POMDPTools.Sim}(undef, NUM_SIM)
greedy_sims = Vector{POMDPTools.Sim}(undef, NUM_SIM)

#@threads for sim_index in 1:NUM_SIM
for sim_index in 1:NUM_SIM
    dgp_rng = Random.MersenneTwister()

    pftdpw_dgp = DGP(dgp_hyperparam, dgp_rng, NUM_PROGRAMS)
    greedy_dgp = deepcopy(pftdpw_dgp)

    pftdpw_mdp = KBanditFundingMDP{SeparateImplementEvalAction, ExponentialUtilityModel}(
        util_model,
        0.95,
        50,
        inference_hyperparam,
        pftdpw_dgp,
        explore_only_actionset_factory
    )

    pftdpw_pomdp = KBanditFundingPOMDP{SeparateImplementEvalAction, ExponentialUtilityModel, FullBayesianBelief{TuringModel}}(pftdpw_mdp)

    particle_updater = MultiBootstrapFilter(pftdpw_pomdp, NUM_FILTER_PARTICLES, bayes_updater)
    bayes_b = initialbelief(pftdpw_pomdp)
    particle_b = POMDPs.initialize_belief(particle_updater, bayes_b)

    belief_mdp = MCTS.GenerativeBeliefMDP(deepcopy(pftdpw_pomdp), particle_updater)
    pftdpw_planner = POMDPs.solve(dpfdpw_solver, belief_mdp)

    greedy_mdp = KBanditFundingMDP{ImplementOnlyAction, ExponentialUtilityModel}(
        util_model,
        0.95,
        50,
        inference_hyperparam,
        greedy_dgp,
        select_subset_actionset_factory;
        curr_state = pftdpw_mdp.curr_state
    )

    greedy_pomdp = KBanditFundingPOMDP{ImplementOnlyAction, ExponentialUtilityModel, FullBayesianBelief{TuringModel}}(greedy_mdp, initialbelief(pftdpw_pomdp))

    greedy_policy = POMDPs.solve(greedy_solver, greedy_pomdp)

    greedy_sim_rng = Random.MersenneTwister()
    pftdpw_sim_rng = copy(greedy_sim_rng)

    pftdpw_sims[sim_index] = POMDPSimulators.Sim(pftdpw_pomdp, pftdpw_planner, particle_updater, rng = greedy_sim_rng, max_steps = NUM_SIM_STEPS)
    greedy_sims[sim_index] = POMDPSimulators.Sim(greedy_pomdp, greedy_policy, bayes_updater, rng = pftdpw_sim_rng, max_steps = NUM_SIM_STEPS)
end

@everywhere function get_sim_data(sim::POMDPTools.Sim, hist::POMDPTools.SimHistory)
    actions = collect(POMDPSimulators.action_hist(hist)) 
    beliefs = collect(POMDPSimulators.belief_hist(hist))

    return (
        action = actions, 
        actual_reward = collect(POMDPSimulators.reward_hist(hist)),
        belief = beliefs,
        expected_reward = [expectedutility(rewardmodel(sim.pomdp), b, a) for (b, a) in zip(beliefs, actions)],

        total_undiscounted_actual_reward = POMDPSimulators.undiscounted_reward(hist)
    )
end

greedy_sim_data = POMDPTools.run_parallel(get_sim_data, greedy_sims; show_progress = true)
pftdpw_sim_data = POMDPTools.run_parallel(get_sim_data, pftdpw_sims; show_progress = true) 