module FundingPOMDPs

import Combinatorics
import Random
import Printf
import DynamicPPL
import StatsBase

using Statistics

using POMDPTools
using POMDPs
using POMDPLinter
using Distributions
using Parameters
using Turing
using DataFrames
using Pipe

include("abstract.jl")
include("SimDGP.jl")
include("reward.jl")
include("states.jl")
include("actions.jl")
include("reward.jl")
include("observations.jl")
include("beliefs.jl")
include("solvers.jl")

export StudyDataset, Hyperparam, sim_model
export AbstractDGP, AbstractProgramDGP, AbstractState 
export DGP, ProgramDGP, CausalState
export AbstractEvalObservation
export EvalObservation
export AbstractRewardModel 
export ExponentialUtilityModel, expectedutility
export AbstractFundingAction 
export ImplementEvalAction
export AbstractActionSet, AbstractFundingAction
export KBanditFundingMDP, KBanditFundingPOMDP, KBanditActionSet
#export discount, isterminal, transition, actions, reward, observation, initialstate, rand 
export numprograms, initialbelief, rewardmodel
export FullBayesianBelief, FullBayesianUpdater
export BayesianGreedySolver, BayesianGreedyPlanner
export TuringModel

struct KBanditActionSet <: AbstractActionSet
    actions::Vector{ImplementEvalAction}
end

function KBanditActionSet(nprograms, nimplement) 
    actionlist = map(Combinatorics.combinations(1:nprograms, nimplement)) do programidx
        ImplementEvalAction(programidx)
    end

    push!(actionlist, ImplementEvalAction()) # Add a do-nothing action

    return KBanditActionSet(actionlist)
end

numactions(as::KBanditActionSet) = length(as.actions)

Base.iterate(as::KBanditActionSet) = iterate(as.actions)
Base.iterate(as::KBanditActionSet, n) = iterate(as.actions, n)

function Base.rand(rng::Random.AbstractRNG, as::KBanditActionSet) 
    actid = Base.rand(rng, 1:numactions(as))

    as.actions[actid] 
end

mutable struct KBanditFundingMDP{S <: AbstractState, A <: AbstractFundingAction, R <: AbstractRewardModel} <: MDP{S, A}
    rewardmodel::R
    discount::Float64
    nimplement::Int64
    studysamplesize::Int64
    dgp::AbstractDGP
    rng::Random.AbstractRNG
    actionset::AbstractActionSet

    initstate::S
end

function KBanditFundingMDP{S, A, R}(r::R, d::Float64, ni::Int64, ss::Int64, dgp::AbstractDGP, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {S, A, R <: AbstractRewardModel}
    ni <= numprograms(dgp) || throw(ArgumentError("number of programs to implement greater than number of programs"))

    initstate = Base.rand(rng, dgp)

    return KBanditFundingMDP{S, A, R}(r, d, ni, ss, dgp, rng, KBanditActionSet(numprograms(dgp), ni), initstate)
end 

mdp(m::KBanditFundingMDP) = m 

struct KBanditFundingPOMDP{S <: AbstractState, A <: AbstractFundingAction, O <: AbstractEvalObservation, R <: AbstractRewardModel, B <: AbstractBelief} <: POMDP{S, A, O}
    mdp::KBanditFundingMDP{S, A, R}

    data::Vector{Vector{StudyDataset}}
    initbelief::B
end

function KBanditFundingPOMDP{S, A, O, R, B}(mdp::KBanditFundingMDP{S, A, R}, data::Vector{Vector{StudyDataset}}) where {S, A, O, R <: AbstractRewardModel, B <: AbstractBelief} 
    return KBanditFundingPOMDP{S, A, O, R, B}(mdp, data, B(data, hyperparam(mdp.dgp)))
end

function KBanditFundingPOMDP{S, A, O, R, B}(mdp::KBanditFundingMDP{S, A, R}) where {S, A, O, R <: AbstractRewardModel, B <: AbstractBelief}
    initdatasets = Vector{Vector{StudyDataset}}(undef, numprograms(mdp.initstate))

    for (pid, ds) in getdatasets(Base.rand(mdp.rng, mdp.initstate, mdp.studysamplesize))
        initdatasets[pid] = [ds] 
    end
    
    return KBanditFundingPOMDP{S, A, O, R, B}(mdp, initdatasets)
end 

function KBanditFundingPOMDP{S, A, O, R, B}(r::R, d::Float64, ni::Int64, ss::Int64, dgp::AbstractDGP, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {S, A, O, R <: AbstractRewardModel, B <: AbstractBelief}
    mdp = KBanditFundingMDP{S, A, R}(r, d, ni, ss, dgp, rng)

    return KBanditFundingPOMDP{S, A, O, R, B}(mdp)
end

const KBanditFundingProblem{S, A, O, R, B} = Union{KBanditFundingPOMDP{S, A, O, R, B}, KBanditFundingMDP{S, A, R}}

mdp(m::KBanditFundingPOMDP) = m.mdp

hyperparam(m::KBanditFundingProblem) = hyperparam(mdp(m).dgp)

numprograms(m::KBanditFundingProblem) = numprograms(mdp(m).dgp)

POMDPs.discount(m::KBanditFundingProblem) = mdp(m).discount

POMDPs.isterminal(m::KBanditFundingProblem) = false 

POMDPs.transition(m::KBanditFundingProblem, ::Any, ::Any) = CausalStateDistribution(mdp(m).dgp) 

POMDPs.actions(m::KBanditFundingProblem) = mdp(m).actionset
POMDPs.actions(m::KBanditFundingProblem, ::Any) = POMDPs.actions(m)

function POMDPs.reward(m::KBanditFundingProblem{S, A, O, R, B}, s::S, a::A) where {S, A, O, R, B}  
    expectedreward = 0

    for (i, p) in s.programstates
        expectedreward += expectedutility(mdp(m).rewardmodel, implements(a, i) ? p.μ + p.τ : p.μ, p.σ)
    end

    return expectedreward
end

rewardmodel(m::KBanditFundingProblem) = mdp(m).rewardmodel

POMDPs.initialstate(m::KBanditFundingMDP) = POMDPTools.Deterministic(m.initstate) # CausalStateDistribution(mdp(m).dgp) 
POMDPs.initialstate(m::KBanditFundingPOMDP) = CausalStateDistribution(mdp(m).dgp) 

function POMDPs.observation(pomdp::KBanditFundingPOMDP{S, A, O, R, B}, s::S, a::A, sp::S) where {S, A, O, R, B}
    return MultiStudySampleDistribution(Dict(pid => StudySampleDistribution(getprogramstate(s, pid), pomdp.mdp.studysamplesize) for pid in get_evaluated_program_ids(a)))
end

initialbelief(m::KBanditFundingPOMDP) = m.initbelief 

end # module