include("FundingPOMDPs.jl")
using .FundingPOMDPs

using DataFrames, StatsBase

import Random
import POMDPs, POMDPTools, POMDPSimulators
import ParticleFilters
import Base: rand, show
import Turing

test_hyperparam = Hyperparam(mu_sd = 1.0, tau_mean = 0.1, tau_sd = 0.25, sigma_sd = 1.0, eta_sd = [0.1, 0.1, 0.1])

dgp = DGP(test_hyperparam, Random.GLOBAL_RNG, 10)
mdp = KBanditFundingMDP{ImplementEvalAction, ExponentialUtilityModel}(
    ExponentialUtilityModel(1.0),
    0.95,
    1,
    50,
    test_hyperparam,
    dgp
 )

pomdp = KBanditFundingPOMDP{ImplementEvalAction, ExponentialUtilityModel, FullBayesianBelief{TuringModel}}(mdp)

fbb = initialbelief(pomdp)

#bsf = ParticleFilters.BootstrapFilter(pomdp, 1000)
bsf = MultiBootstrapFilter(pomdp, 1000)

gs = BayesianGreedySolver()
gp = POMDPs.solve(gs, pomdp)

hr = POMDPSimulators.HistoryRecorder(max_steps = 10)
history = POMDPs.simulate(hr, pomdp, gp, bsf, fbb)

[h.a for h in history]
[h.bp for h in history]

[expectedutility(pomdp.mdp.rewardmodel, pomdp.mdp.dgp, a) for a in POMDPs.actions(pomdp)]
[expectedutility(pomdp.mdp.rewardmodel, history[10].bp, a) for a in POMDPs.actions(pomdp)]

last_b = fbb
up = POMDPs.updater(gp)
hr2 = POMDPSimulators.HistoryRecorder(max_steps = 10)
history2 = POMDPs.simulate(hr2, pomdp, gp, up, fbb)

[h.a for h in history2]
[expectedutility(pomdp.mdp.rewardmodel, history2[10].bp, a) for a in POMDPs.actions(pomdp)]

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

#rand(Random.GLOBAL_RNG, fbb)


# r = pomdp.mdp.rewardmodel
# d = fbb.posterior_samples[1]
# expectedutility(r, d.μ_toplevel, d.σ_toplevel)

#[POMDPs.value(gp, fbb, a) for a in POMDPs.actions(pomdp)]
#ba = POMDPs.action(gp, fbb)

#=

last_b = fbb

for (a, o, b, bp) in POMDPSimulators.stepthrough(pomdp, gp, up, fbb, "a, o, b, bp", max_steps = 10)
    @show a, o
    @show POMDPs.value(gp, b)
    @show POMDPs.value(gp, bp)

    global last_b = bp
end

[expectedutility(pomdp.mdp.rewardmodel, last_b, a) for a in POMDPs.actions(pomdp)]

pomdp.mdp.dgp.programdgps

pomdp.mdp.dgp.programdgps[6]
last_b.progbeliefs[3]
last_b.datasets[3]

expectedutility(pomdp.mdp.rewardmodel, pomdp.mdp.dgp.programdgps[3], true)
expectedutility(pomdp.mdp.rewardmodel, last_b.progbeliefs[3], true)

[expectedutility(pomdp.mdp.rewardmodel, pomdp.mdp.dgp.programdgps[pid], true) for pid in 1:10] 
[expectedutility(pomdp.mdp.rewardmodel, last_b.progbeliefs[pid], true) for pid in 1:10]
=#

#=
ds = last_b.datasets[6]
bm = sim_model(test_hyperparam, ds[:])
c = Turing.sample(bm, Turing.NUTS(), Turing.MCMCThreads(), 500, 4)
mean(DataFrame(c).τ_toplevel)
=#