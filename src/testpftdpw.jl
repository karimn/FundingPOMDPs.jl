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
const NUM_SIM = 3 
const NUM_SIM_STEPS = 5 
const NUM_FILTER_PARTICLES = 1_000

dgp_hyperparam = Hyperparam(mu_sd = 1.0, tau_mean = 0.1, tau_sd = 0.25, sigma_sd = 1.0, eta_sd = [0.1, 0.1, 0.1])
inference_hyperparam = Hyperparam(mu_sd = 2.0, tau_mean = 0.0, tau_sd = 0.5, sigma_sd = 4.0, eta_sd = [0.2, 0.2, 0.2])

util_model = ExponentialUtilityModel(1.0)

#sep_impl_eval_actionset_factory = SeparateImplementAndEvalActionSetFactory(NUM_PROGRAMS)
select_subset_actionset_factory = SelectProgramSubsetActionSetFactory(NUM_PROGRAMS, 1)
explore_only_actionset_factory = ExploreOnlyActionSetFactory(NUM_PROGRAMS, 1, 1, util_model)

dpfdpw_solver = MCTS.DPWSolver(
    depth = 10,
    exploration_constant = 0.0,
    n_iterations = 200,
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

@threads for sim_index in 1:NUM_SIM
#for sim_index in 1:NUM_SIM
    dgp = DGP(dgp_hyperparam, RNG, NUM_PROGRAMS)

    pftdpw_mdp = KBanditFundingMDP{SeparateImplementEvalAction, ExponentialUtilityModel}(
        util_model,
        0.95,
        50,
        inference_hyperparam,
        dgp,
        #sep_impl_eval_actionset_factory 
        explore_only_actionset_factory
    )

    pftdpw_pomdp = KBanditFundingPOMDP{SeparateImplementEvalAction, ExponentialUtilityModel, FullBayesianBelief{TuringModel}}(pftdpw_mdp)

    particle_updater = MultiBootstrapFilter(pftdpw_pomdp, NUM_FILTER_PARTICLES, bayes_updater)
    bayes_b = initialbelief(pftdpw_pomdp)
    particle_b = POMDPs.initialize_belief(particle_updater, bayes_b)

    belief_mdp = MCTS.GenerativeBeliefMDP(pftdpw_pomdp, particle_updater)
    pftdpw_planner = POMDPs.solve(dpfdpw_solver, belief_mdp)

    pftdpw_sims[sim_index] = POMDPSimulators.Sim(pftdpw_pomdp, pftdpw_planner, particle_updater, rng = RNG, max_steps = NUM_SIM_STEPS)

    greedy_mdp = KBanditFundingMDP{ImplementOnlyAction, ExponentialUtilityModel}(
        util_model,
        0.95,
        50,
        inference_hyperparam,
        dgp,
        select_subset_actionset_factory
    )

    greedy_pomdp = KBanditFundingPOMDP{ImplementOnlyAction, ExponentialUtilityModel, FullBayesianBelief{TuringModel}}(greedy_mdp, initialbelief(pftdpw_pomdp))

    greedy_policy = POMDPs.solve(greedy_solver, greedy_pomdp)

    greedy_sims[sim_index] = POMDPSimulators.Sim(greedy_pomdp, greedy_policy, bayes_updater, rng = RNG, max_steps = NUM_SIM_STEPS)
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

#=
s = rand(RNG, POMDPs.initialstate(mdp))
a = POMDPs.action(pftdpw, particle_b)

sp, o = @POMDPs.gen(:sp, :o)(pomdp, s, a, RNG)

bayes_bp = POMDPs.update(bayes_updater, bayes_b, a, o)


[expectedutility(pomdp.mdp.rewardmodel, pomdp.mdp.dgp, a) for a in POMDPs.actions(pomdp)]

gs = BayesianGreedySolver()
gp = POMDPs.solve(gs, pomdp)

hr = POMDPSimulators.HistoryRecorder(max_steps = 10)
history = POMDPs.simulate(hr, pomdp, gp, bsf, b0)

[expectedutility(pomdp.mdp.rewardmodel, history[1].bp, a) for a in POMDPs.actions(pomdp)]
=#