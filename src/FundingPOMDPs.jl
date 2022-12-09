module FundingPOMDPs

import Combinatorics
import Random
import Printf

using Statistics

using POMDPTools
using POMDPs
using POMDPLinter
using Distributions

include("SimDGP.jl")

include("abstract.jl")
include("states.jl")
include("actions.jl")
include("reward.jl")
include("observations.jl")

export StudyDataset, Hyperparam
export AbstractDGP, AbstractProgramDGP, AbstractState 
export DGP, ProgramDGP, CausalState
export AbstractEvalObservation
export EvalObservation
export AbstractRewardModel 
export ExponentialUtilityModel
export AbstractFundingAction 
export ImplementEvalAction
export AbstractActionSet
export KBanditFundingMDP, KBanditFundingPOMDP, KBanditActionSet
#export discount, isterminal, transition, actions, reward, observation, initialstate, rand 
export numprograms

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

function Base.rand(rng::Random.AbstractRNG, as::KBanditActionSet) 
    actid = Base.rand(rng, 1:numactions(as))

    as.actions[actid] 
end

struct KBanditFundingMDP{S <: AbstractState, A <: AbstractFundingAction, R <: AbstractRewardModel} <: MDP{S, A}
    rewardmodel::R
    discount::Float64
    nimplement::Int64
    studysamplesize::Int64
    dgp::AbstractDGP
    actionset::AbstractActionSet
end

function KBanditFundingMDP{S, A, R}(r::R, d::Float64, ni::Int64, ss::Int64, dgp::AbstractDGP) where {S, A, R <: AbstractRewardModel}
    ni <= numprograms(dgp) || throw(ArgumentError("number of programs to implement greater than number of programs"))

    KBanditFundingMDP{S, A, R}(r, d, ni, ss, dgp, KBanditActionSet(numprograms(dgp), ni))
end 

mdp(m::KBanditFundingMDP) = m 

struct KBanditFundingPOMDP{S <: AbstractState, A <: AbstractFundingAction, O <: AbstractEvalObservation, R <: AbstractRewardModel} <: POMDP{S, A, O}
    mdp::KBanditFundingMDP{S, A, R}
end

function KBanditFundingPOMDP{S, A, O, R}(r::R, d::Float64, ni::Int64, ss::Int64, dgp::AbstractDGP) where {S, A, O, R <: AbstractRewardModel}
    KBanditFundingPOMDP{S, A, O, R}(KBanditFundingMDP{S, A, R}(r, d, ni, ss, dgp, KBanditActionSet(numprograms(dgp), ni)))
end

const KBanditFundingProblem{S, A, O, R} = Union{KBanditFundingPOMDP{S, A, O, R}, KBanditFundingMDP{S, A, R}}

mdp(m::KBanditFundingPOMDP) = m.mdp

numprograms(m::KBanditFundingProblem) = numprograms(mdp(m).dgp)

POMDPs.discount(m::KBanditFundingProblem) = mdp(m).discount

POMDPs.isterminal(m::KBanditFundingProblem) = false 

POMDPs.transition(m::KBanditFundingProblem, ::Any, ::Any) = CausalStateDistribution(mdp(m).dgp) #POMDPTools.Deterministic(s)

POMDPs.actions(m::KBanditFundingProblem) = mdp(m).actionset
POMDPs.actions(m::KBanditFundingProblem, ::Any) = POMDPs.actions(m)

function POMDPs.reward(m::KBanditFundingProblem{S, A, O, R}, s::S, a::A) where {S, A, O, R}  
    expectedreward = 0

    for (i, p) in s.programstates
        expectedreward += expectedutility(mdp(m).rewardmodel, implements(a, i) ? p.μ + p.τ : p.μ, p.σ)
    end

    return expectedreward
end

POMDPs.initialstate(m::KBanditFundingProblem) = CausalStateDistribution(mdp(m).dgp) 

function POMDPs.observation(pomdp::KBanditFundingPOMDP{S, A, O, R}, s::S, a::A, sp::S) where {S, A, O, R}
    return MultiStudySampleDistribution(Dict(pid => StudySampleDistribution(getprogramstate(s, pid), pomdp.mdp.studysamplesize) for pid in get_evaluated_program_ids(a)))
end

end # module