include("FundingPOMDPs.jl")
using .FundingPOMDPs

using DataFrames, StatsBase

import Random
import POMDPs, POMDPTools, POMDPSimulators
import ParticleFilters
import Base: rand, show
import Turing

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

#=

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

bayes_model = sim_model(test_hyperparam, [pomdp.data[1][1]])
c = Turing.sample(bayes_model, Turing.NUTS(), Turing.MCMCThreads(), 500, 4)
=#

fbb = initialbelief(pomdp)
#rand(Random.GLOBAL_RNG, fbb)

gs = BayesianGreedySolver()
gp = POMDPs.solve(gs, pomdp)
up = POMDPs.updater(gp)

# r = pomdp.mdp.rewardmodel
# d = fbb.posterior_samples[1]
# expectedutility(r, d.μ_toplevel, d.σ_toplevel)

#[POMDPs.value(gp, fbb, a) for a in POMDPs.actions(pomdp)]
#ba = POMDPs.action(gp, fbb)

last_b = fbb

for (a, o, b, bp) in POMDPSimulators.stepthrough(pomdp, gp, up, fbb, "a, o, b, bp", max_steps = 10)
    @show a, o
    @show POMDPs.value(gp, b)
    @show POMDPs.value(gp, bp)

    global last_b = bp
end

[expectedutility(pomdp.mdp.rewardmodel, last_b, a) for a in POMDPs.actions(pomdp)]
[expectedutility(pomdp.mdp.rewardmodel, pomdp.mdp.dgp, a) for a in POMDPs.actions(pomdp)]

pomdp.mdp.dgp.programdgps[4]
last_b.progbeliefs[4]
last_b.datasets[4]

ds = last_b.datasets[4]
bm = sim_model(test_hyperparam, ds[1:1])
c = Turing.sample(bm, Turing.NUTS(), Turing.MCMCThreads(), 500, 4)
mean(DataFrame(c).τ_toplevel)