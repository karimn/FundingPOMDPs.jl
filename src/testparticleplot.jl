include("FundingPOMDPs.jl")
using .FundingPOMDPs

using DataFrames, StatsBase
using Plots, StatsPlots

import Random
import POMDPs, POMDPTools, POMDPSimulators, POMDPPolicies
import ParticleFilters
import Base: rand, show
import Turing

test_hyperparam = Hyperparam(mu_sd = 1.0, tau_mean = 0.1, tau_sd = 0.25, sigma_sd = 1.0, eta_sd = [0.1, 0.1, 0.1])

dgp = DGP(test_hyperparam, Random.GLOBAL_RNG, 1, ProgramDGP)
mdp = KBanditFundingMDP{ImplementEvalAction, ExponentialUtilityModel}(
    ExponentialUtilityModel(1.0),
    0.95,
    1,
    50,
    test_hyperparam,
    dgp
 )

 struct OneActionPolicy <: POMDPs.Policy
    pomdp::KBanditFundingPOMDP
 end

POMDPs.action(p::OneActionPolicy, s::CausalState) = POMDPs.actions(p.pomdp)[1]

pomdp = KBanditFundingPOMDP{ImplementEvalAction, ExponentialUtilityModel, FullBayesianBelief{TuringModel}}(mdp)
policy = OneActionPolicy(pomdp)
bsf = MultiBootstrapFilter(pomdp, 10_000)
b0 = initialbelief(pomdp)

function plot_belief(b, s, sp = missing)
    bdf = convert(DataFrame, b)
    p = @df bdf scatter(:μ, :σ) 
    scatter!(p, [s.programstates[1].μ], [s.programstates[1].σ])
    sp === missing || scatter!(p, [sp.programstates[1].μ], [sp.programstates[1].σ])
    xlims!(p, (-1.5, 0))
    ylims!(p, (0.5, 1))

    return p
end

function make_particle_anim(pomdp, updater, b0, policy)
    local b = POMDPs.initialize_belief(updater, b0)
    local s = rand(POMDPs.initialstate(pomdp))

    local a = POMDPs.action(policy, s)

    anim = Animation()

    frame(anim, plot_belief(b, s))

    for i in 1:20
        sp, o, r = POMDPs.@gen(:sp, :o, :r)(pomdp, s, a)
        b = POMDPs.update(updater, b, a, o)

        frame(anim, plot_belief(b, s, sp))

        s = sp
    end 

    return anim
end

anim = make_particle_anim(pomdp, bsf, b0, policy) 

gif(anim, "anim.gif", fps = 1)