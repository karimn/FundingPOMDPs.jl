include("FundingPOMDPs.jl")
using .FundingPOMDPs

import Random
import POMDPs
import Base: rand, show

test_hyperparam = Hyperparam(mu_sd = 1.0, tau_mean = 0.1, tau_sd = 0.25, sigma_sd = 1.0, eta_sd = [0.1, 0.1, 0.1])

dgp = DGP(test_hyperparam, Random.GLOBAL_RNG, 10, ProgramDGP)
mdp = KBanditFundingMDP{CausalState, ImplementEvalAction, ExponentialUtilityModel}(
    ExponentialUtilityModel(1.0),
    0.95,
    1,
    50,
    dgp 
)

pomdp = KBanditFundingPOMDP{CausalState, ImplementEvalAction, EvalObservation, ExponentialUtilityModel}(mdp)

inst = POMDPs.initialstate(pomdp)
st = rand(Random.GLOBAL_RNG, inst)