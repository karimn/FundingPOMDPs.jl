include("FundingPOMDPs.jl")
using .FundingPOMDPs

import Random
import POMDPs, POMDPTools, POMDPSimulators
import ParticleFilters
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

randpolicy = POMDPTools.RandomPolicy(pomdp)
bsf = ParticleFilters.BootstrapFilter(pomdp, 1_000)

for (s, a, o) in POMDPSimulators.stepthrough(pomdp, randpolicy, "s, a, o", max_steps = 100)
    #@show s, a, o
    @show a
end

stept = collect(POMDPSimulators.stepthrough(pomdp, randpolicy, bsf, inst, st, "b, bp", max_steps = 1))
values(ParticleFilters.probdict(stept[1].b))

for (s, a, o, b, bp) in POMDPSimulators.stepthrough(pomdp, randpolicy, bsf, inst, st, "s, a, o, b, bp", max_steps = 1000)
    @show s, a, o
    #@show a
    #@show b bp
end

POMDPs.updater(randpolicy)