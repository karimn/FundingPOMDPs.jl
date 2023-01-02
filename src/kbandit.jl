
struct KBanditFundingMDP{A <: AbstractFundingAction, R <: AbstractRewardModel} <: MDP{CausalState, A}
    rewardmodel::R
    discount::Float64
    studysamplesize::Int64
    inference_hyperparam::Hyperparam
    dgp::AbstractDGP
    actionset_factory::AbstractActionSetFactory{A}
    rng::Random.AbstractRNG

    curr_state::CausalState
end

function KBanditFundingMDP{A, R}(r::R, d::Float64, ss::Int64, inference_hyperparam::Hyperparam, dgp::AbstractDGP, asf::AbstractActionSetFactory{A}, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {A, R <: AbstractRewardModel}
    curr_state = Base.rand(rng, dgp)

    return KBanditFundingMDP{A, R}(r, d, ss, inference_hyperparam, dgp, asf, rng, curr_state)
end 

mdp(m::KBanditFundingMDP) = m 

struct KBanditFundingPOMDP{A <: AbstractFundingAction, R <: AbstractRewardModel, B <: AbstractBelief} <: POMDP{CausalState, A, EvalObservation}
    mdp::KBanditFundingMDP{A, R}
    curr_belief::B
end

function KBanditFundingPOMDP{A, R, B}(mdp::KBanditFundingMDP{A, R}, data::Vector{Vector{StudyDataset}}) where {A <: AbstractFundingAction, R <: AbstractRewardModel, B <: AbstractBelief} 
    return KBanditFundingPOMDP{A, R, B}(mdp, B(data, mdp.inference_hyperparam, mdp.rng))
end

function KBanditFundingPOMDP{A, R, B}(mdp::KBanditFundingMDP{A, R}) where {A <: AbstractFundingAction, R <: AbstractRewardModel, B <: AbstractBelief}
    initdatasets = Vector{Vector{StudyDataset}}(undef, numprograms(mdp.curr_state))

    for (pid, ds) in getdatasets(Base.rand(mdp.rng, mdp.curr_state, mdp.studysamplesize))
        initdatasets[pid] = [ds] 
    end
    
    return KBanditFundingPOMDP{A, R, B}(mdp, initdatasets)
end 

function KBanditFundingPOMDP{A, R, B}(r::R, d::Float64, ss::Int64, dgp::AbstractDGP, asf::AbstractActionSetFactory{A}, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {A, R <: AbstractRewardModel, B <: AbstractBelief}
    mdp = KBanditFundingMDP{A, R}(r, d, ss, dgp, asf, rng)

    return KBanditFundingPOMDP{A, R, B}(mdp)
end

const KBanditFundingProblem{A, R, B} = Union{KBanditFundingPOMDP{A, R, B}, KBanditFundingMDP{A, R}}

struct ProgramBanditWrapper{A <: AbstractFundingAction, R <: AbstractRewardModel, B <: AbstractBelief} <: POMDPs.POMDP{ProgramCausalState, A, ProgramEvalObservation}
    wrapped_problem::KBanditFundingPOMDP{A, R, B}
    pid::Int
end

programbandits(m::KBanditFundingPOMDP) = [ProgramBanditWrapper(m, i) for i in 1:numprograms(m)] 

mdp(m::KBanditFundingPOMDP) = m.mdp

hyperparam(m::KBanditFundingProblem) = mdp(m).inference_hyperparam

numprograms(m::KBanditFundingProblem) = numprograms(mdp(m).dgp)

rewardmodel(m::KBanditFundingProblem) = mdp(m).rewardmodel

initialbelief(m::KBanditFundingPOMDP) = m.curr_belief 

POMDPs.discount(m::KBanditFundingProblem) = mdp(m).discount

POMDPs.isterminal(m::KBanditFundingProblem) = false 
POMDPs.isterminal(m::POMDPTools.GenerativeBeliefMDP, b::AbstractBelief) = false  # So we don't have to look at the entire support

POMDPs.transition(m::KBanditFundingProblem, s::CausalState, a::AbstractFundingAction) = transition(s, a)  #CausalStateDistribution(mdp(m).dgp) 

POMDPs.transition(m::ProgramBanditWrapper, s::ProgramCausalState, a::AbstractFundingAction) = transition(s, a)  #CausalStateDistribution(mdp(m).dgp) 

POMDPs.actions(m::KBanditFundingProblem) = actions(mdp(m).actionset_factory)
POMDPs.actions(m::KBanditFundingProblem, s::CausalState) = actions(mdp(m).actionset_factory, s)
POMDPs.actions(m::KBanditFundingPOMDP{A, R, B}, b::AbstractBelief) where {A <: AbstractFundingAction, R, B <: AbstractBelief} = actions(mdp(m).actionset_factory, b)

POMDPs.reward(m::KBanditFundingProblem{A, R, B}, s::CausalState, a::A) where {A, R, B} = expectedutility(rewardmodel(m), s, a)

POMDPs.initialstate(m::KBanditFundingMDP) = POMDPTools.Deterministic(m.curr_state) # CausalStateDistribution(mdp(m).dgp) 
POMDPs.initialstate(m::KBanditFundingPOMDP) = initialbelief(m) # CausalStateDistribution(mdp(m).dgp) 

function POMDPs.observation(pomdp::KBanditFundingPOMDP{A, R, B}, s::CausalState, a::A, sp::CausalState) where {A, R, B}
    return MultiStudySampleDistribution(Dict(pid => StudySampleDistribution(getprogramstate(s, pid), pomdp.mdp.studysamplesize) for pid in get_evaluated_program_ids(a)))
end

function POMDPs.observation(pomdp::ProgramBanditWrapper{A, R, B}, s::ProgramCausalState, a::A, sp::ProgramCausalState) where {A, R, B}
    @assert evaluates(a, getprogramid(s))

    return StudySampleDistribution(s, pomdp.wrapped_problem.mdp.studysamplesize)
end
