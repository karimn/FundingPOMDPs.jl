module KBanditFundingPOMDP

import Combinatorics
import Random

using POMDPTools
using POMDPs
using POMDPLinter
using Distributions

include("SimDGP.jl")

using .SimDGP

include("states.jl")
include("actions.jl")
include("reward.jl")
include("observations.jl")

struct KBanditActionSet{T <: AbstractProgramState} <: AbstractActionSet{T}
    actions::Vector{ImplementEvalAction{T}}
end

function KBanditActionSet{T}(programs, nimplement) where {T}
    actionlist = map(Combinatorics.combinations(1:length(programs), nimplement)) do programidx
        ImplementEvalAction{T}(programidx, programs[programidx])
    end

    push!(actionlist, ImplementEvalAction{T}()) # Add a do-nothing action

    return KBanditActionSet{T}(actionlist)
end

numactions(as::KBanditActionSet{T}) where {T} = length(as.implement_eval_programs)

Base.iterate(as::KBanditActionSet{T}) where {T} = iterate(as.actions)

Base.rand(rng::Random.AbstractRNG, as::KBanditActionSet{T}) where {T} = as.implement_eval_programs[rang(rng, 1:length(numactions(as)))] 

struct KBanditFundingPOMDP{S <: AbstractState, A <: AbstractFundingAction, O <: AbstractEvalObservation, R <: AbstractRewardModel} <: POMDP{S, A, O}
    rewardmodel::R
    discount::Float64
    nimplement::Int64
    studysamplesize::Int64
    actualstate::S
    actionset::A
end

function KBanditFundingPOMDP{S, A, O, R}(r, d, ni, ss, as) where {S, A, O, R}
    ni <= numprograms(as) || throw(ArgumentError("number of programs to implement greater than number of programs"))

    new(r, d, ni, ss, as, KBanditActionSet(as.programstates, ni))
end

numprograms(pomdp::KBanditFundingPOMDP{S, A, O, R}) where {S, A, O, R} = numprograms(pomdp.actualstate)

POMDPs.discount(pomdp::KBanditFundingPOMDP{S, A, O, R}) where {S, A, O, R} = pomdp.discount

POMDPs.isterminal(pomdp::KBanditFundingPOMDP{S, A, O, R}) where {S, A, O, R} = false 

POMDPs.transition(::KBanditFundingPOMDP{S, A, O, R}, s::S, ::A) where {S, A, O, R} = POMDPTools.Deterministic(s)

POMDPs.actions(pomdp::KBanditFundingPOMDP{S, A, O, R}, s::S) where {S, A, O, R} = pomdp.actionset

function POMDPs.reward(pomdp::KBanditFundingPOMDP{S, A, O, R}, s::S, a::A) where {S, A, O, R}  
    expectedreward = 0

    for (i, p) in s.programstates
        if a.implements(i)
            μ = p.μ + p.τ
            σ = sqrt(p.σ^2 + p.η[2]^2)
        else
            μ = p.μ
            σ = sqrt(p.σ^2 + p.η[1]^2)
        end

        expectedreward += expectedutility(pomdp.rewardmodel, μ, σ)
    end

    return expectedreward
end

function POMDPs.observation(pomdp::KBanditFundingPOMDP{S, A, O, R}, a::A, sp::S) where {S, A, O, R}
    return MultiStudySampleDistribution{StudySampleDistribution{S}}([StudySampleDistribution{S}(program, pomdp.studysamplesize) for program in get_evaluated_programs(a)])
end

const KBanditState = State{ProgramState}

end # module