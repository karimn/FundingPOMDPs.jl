include("FundingPOMDPs.jl")
using .FundingPOMDPs

using DataFrames, StatsBase

import Random
import POMDPs, POMDPTools, POMDPSimulators
import ParticleFilters
import MCTS
import Base: rand, show
import Turing

rng = Random.MersenneTwister(123)

test_hyperparam = Hyperparam(mu_sd = 1.0, tau_mean = 0.1, tau_sd = 0.25, sigma_sd = 1.0, eta_sd = [0.1, 0.1, 0.1])

dgp = DGP(test_hyperparam, rng, 10)
mdp = KBanditFundingMDP{ImplementEvalAction, ExponentialUtilityModel}(
    ExponentialUtilityModel(1.0),
    0.95,
    1,
    50,
    test_hyperparam,
    dgp
 )

pomdp = KBanditFundingPOMDP{ImplementEvalAction, ExponentialUtilityModel, FullBayesianBelief{TuringModel}}(mdp)

bsf = MultiBootstrapFilter(pomdp, 1_000)
b0 = initialbelief(pomdp)
b = POMDPs.initialize_belief(bsf, b0)

solver = MCTS.DPWSolver(
    depth = 10,
    #exploration_constant = 1,
    n_iterations = 1000,
    enable_action_pw = false,
    k_state = 4.5,
    alpha_state = 1/10.0,
    check_repeat_state = false,
    estimate_value = MCTS.RolloutEstimator(MCTS.RandomSolver(rng)),
    rng = rng
)

belief_mdp = MCTS.GenerativeBeliefMDP(deepcopy(pomdp), bsf)
pftdpw = POMDPs.solve(solver, belief_mdp)

a = POMDPs.action(pftdpw, b)

[expectedutility(pomdp.mdp.rewardmodel, pomdp.mdp.dgp, a) for a in POMDPs.actions(pomdp)]

gs = BayesianGreedySolver()
gp = POMDPs.solve(gs, pomdp)

hr = POMDPSimulators.HistoryRecorder(max_steps = 10)
history = POMDPs.simulate(hr, pomdp, gp, bsf, b0)

[expectedutility(pomdp.mdp.rewardmodel, history[1].bp, a) for a in POMDPs.actions(pomdp)]