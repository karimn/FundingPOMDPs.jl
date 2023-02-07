
struct KBanditFundingMDP{A <: AbstractFundingAction} <: MDP{CausalState, A}
    rewardmodel::AbstractRewardModel
    discount::Float64
    studysamplesize::Int64
    actionset_factory::AbstractActionSetFactory{A}
    rng::Random.AbstractRNG

    pre_state::CausalState
end

#=
function KBanditFundingMDP{A}(r::AbstractRewardModel, d::Float64, ss::Int64, dgp::AbstractDGP, asf::AbstractActionSetFactory{A}; rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {A}
    curr_state = Base.rand(rng, dgp)

    return KBanditFundingMDP{A}(r, d, ss, asf, rng, curr_state)
end 
=#

mdp(m::KBanditFundingMDP) = m 

struct KBanditFundingPOMDP{A <: AbstractFundingAction} <: POMDP{CausalState, A, EvalObservation}
    mdp::KBanditFundingMDP{A}
    curr_belief::Belief
end

function KBanditFundingPOMDP{A}(mdp::KBanditFundingMDP{A}, data::Vector{Vector{StudyDataset}}, m::AbstractLearningModel) where {A <: AbstractFundingAction} 
    return KBanditFundingPOMDP{A}(mdp, Belief(data, m))
end

function KBanditFundingPOMDP{A}(mdp::KBanditFundingMDP{A}, belief::Belief, m::AbstractLearningModel) where {A <: AbstractFundingAction} 
    return KBanditFundingPOMDP{A}(mdp, [data(pb) for pb in belief], m)
end

function KBanditFundingPOMDP{A}(mdp::KBanditFundingMDP{A}, m::AbstractLearningModel) where {A <: AbstractFundingAction}
    initdatasets = Vector{Vector{StudyDataset}}(undef, numprograms(mdp))

    for (pid, ds) in getdatasets(Base.rand(mdp.rng, mdp.pre_state, mdp.studysamplesize))
        initdatasets[pid] = [ds] 
    end
    
    return KBanditFundingPOMDP{A}(mdp, initdatasets, m)
end 

function KBanditFundingPOMDP{A}(r::AbstractRewardModel, d::Float64, ss::Int64, dgp::AbstractDGP, asf::AbstractActionSetFactory{A}, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {A}
    mdp = KBanditFundingMDP{A}(r, d, ss, dgp, asf, rng)

    return KBanditFundingPOMDP{A}(mdp)
end

const KBanditFundingProblem{A} = Union{KBanditFundingPOMDP{A}, KBanditFundingMDP{A}}

struct ProgramBanditWrapper{A <: AbstractFundingAction} <: POMDPs.POMDP{ProgramCausalState, A, ProgramEvalObservation}
    wrapped_problem::KBanditFundingPOMDP{A}
    pid::Int
end

programbandits(m::KBanditFundingPOMDP) = [ProgramBanditWrapper(m, i) for i in 1:numprograms(m)] 

mdp(m::KBanditFundingPOMDP) = m.mdp

numprograms(m::KBanditFundingProblem) = numprograms(mdp(m).pre_state)

rewardmodel(m::KBanditFundingProblem) = mdp(m).rewardmodel

initialbelief(m::KBanditFundingPOMDP) = m.curr_belief 

POMDPs.discount(m::KBanditFundingProblem) = mdp(m).discount

POMDPs.isterminal(m::KBanditFundingProblem) = false 
POMDPs.isterminal(m::POMDPTools.GenerativeBeliefMDP, b::AbstractBelief) = false  # So we don't have to look at the entire support

POMDPs.transition(m::KBanditFundingProblem, s::CausalState, a::AbstractFundingAction) = transition(s, a)  
POMDPs.transition(m::ProgramBanditWrapper, s::ProgramCausalState, a::AbstractFundingAction) = transition(s, a) 

POMDPs.actions(m::KBanditFundingProblem) = actions(mdp(m).actionset_factory)
POMDPs.actions(m::KBanditFundingProblem, s::CausalState) = actions(mdp(m).actionset_factory, s)
POMDPs.actions(m::KBanditFundingPOMDP{A}, b::AbstractBelief) where {A <: AbstractFundingAction} = actions(mdp(m).actionset_factory, b)

POMDPs.reward(m::KBanditFundingProblem{A}, s::CausalState, a::A) where {A} = expectedutility(rewardmodel(m), s, a)

POMDPs.initialstate(m::KBanditFundingMDP) = POMDPTools.Deterministic(next_state(mdp(m).pre_state))

# GenerativeBeliefMDP uses this method to figure out the belief type of the POMDP so I can't just return the causal state.
POMDPs.initialstate(m::KBanditFundingPOMDP) = initialbelief(m)

function POMDPs.observation(pomdp::KBanditFundingPOMDP{A}, s::CausalState, a::A, sp::CausalState) where {A}
    return MultiStudySampleDistribution(Dict(pid => StudySampleDistribution(getprogramstate(s, pid), pomdp.mdp.studysamplesize) for pid in get_evaluated_program_ids(a)))
end

function POMDPs.observation(pomdp::ProgramBanditWrapper{A}, s::ProgramCausalState, a::A, sp::ProgramCausalState) where {A}
    @assert evaluates(a, getprogramid(s))

    return StudySampleDistribution(s, pomdp.wrapped_problem.mdp.studysamplesize)
end
