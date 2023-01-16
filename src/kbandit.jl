
struct KBanditFundingMDP{A <: AbstractFundingAction} <: MDP{CausalState, A}
    rewardmodel::AbstractRewardModel
    discount::Float64
    studysamplesize::Int64
    dgp::AbstractDGP
    actionset_factory::AbstractActionSetFactory{A}
    rng::Random.AbstractRNG

    curr_state::CausalState

    function KBanditFundingMDP{A}(r::AbstractRewardModel, d::Float64, ss::Int64, dgp::AbstractDGP, asf::AbstractActionSetFactory{A}; rng::Random.AbstractRNG = Random.GLOBAL_RNG, curr_state::CausalState = Base.rand(rng, dgp)) where {A}
        return new{A}(r, d, ss, dgp, asf, rng, curr_state)
    end 
end


mdp(m::KBanditFundingMDP) = m 

struct KBanditFundingPOMDP{A <: AbstractFundingAction, B <: AbstractBelief} <: POMDP{CausalState, A, EvalObservation}
    mdp::KBanditFundingMDP{A}
    curr_belief::B
end

function KBanditFundingPOMDP{A,B}(mdp::KBanditFundingMDP{A}, data::Vector{Vector{StudyDataset}}, m::AbstractBayesianModel) where {A <: AbstractFundingAction, B <: AbstractBelief} 
    return KBanditFundingPOMDP{A, B}(mdp, B(data, m, mdp.rng))
end

function KBanditFundingPOMDP{A, B}(mdp::KBanditFundingMDP{A}, belief::B, m::AbstractBayesianModel) where {A <: AbstractFundingAction, B <: AbstractBelief} 
    return KBanditFundingPOMDP{A, B}(mdp, [data(pb) for pb in belief], m)
end

function KBanditFundingPOMDP{A, B}(mdp::KBanditFundingMDP{A}, m::AbstractBayesianModel) where {A <: AbstractFundingAction, B <: AbstractBelief}
    initdatasets = Vector{Vector{StudyDataset}}(undef, numprograms(mdp.curr_state))

    for (pid, ds) in getdatasets(Base.rand(mdp.rng, mdp.curr_state, mdp.studysamplesize))
        initdatasets[pid] = [ds] 
    end
    
    return KBanditFundingPOMDP{A, B}(mdp, initdatasets, m)
end 

function KBanditFundingPOMDP{A, B}(r::AbstractRewardModel, d::Float64, ss::Int64, dgp::AbstractDGP, asf::AbstractActionSetFactory{A}, rng::Random.AbstractRNG = Random.GLOBAL_RNG) where {A, B <: AbstractBelief}
    mdp = KBanditFundingMDP{A}(r, d, ss, dgp, asf, rng)

    return KBanditFundingPOMDP{A, B}(mdp)
end

const KBanditFundingProblem{A, B} = Union{KBanditFundingPOMDP{A,B}, KBanditFundingMDP{A}}

struct ProgramBanditWrapper{A <: AbstractFundingAction, B <: AbstractBelief} <: POMDPs.POMDP{ProgramCausalState, A, ProgramEvalObservation}
    wrapped_problem::KBanditFundingPOMDP{A, B}
    pid::Int
end

programbandits(m::KBanditFundingPOMDP) = [ProgramBanditWrapper(m, i) for i in 1:numprograms(m)] 

mdp(m::KBanditFundingPOMDP) = m.mdp

numprograms(m::KBanditFundingProblem) = numprograms(mdp(m).dgp)

rewardmodel(m::KBanditFundingProblem) = mdp(m).rewardmodel

initialbelief(m::KBanditFundingPOMDP) = m.curr_belief 

POMDPs.discount(m::KBanditFundingProblem) = mdp(m).discount

POMDPs.isterminal(m::KBanditFundingProblem) = false 
POMDPs.isterminal(m::POMDPTools.GenerativeBeliefMDP, b::AbstractBelief) = false  # So we don't have to look at the entire support

POMDPs.transition(m::KBanditFundingProblem, s::CausalState, a::AbstractFundingAction) = transition(s, a)  
POMDPs.transition(m::ProgramBanditWrapper, s::ProgramCausalState, a::AbstractFundingAction) = transition(s, a) 

POMDPs.actions(m::KBanditFundingProblem) = actions(mdp(m).actionset_factory)
POMDPs.actions(m::KBanditFundingProblem, s::CausalState) = actions(mdp(m).actionset_factory, s)
POMDPs.actions(m::KBanditFundingPOMDP{A, B}, b::AbstractBelief) where {A <: AbstractFundingAction, B <: AbstractBelief} = actions(mdp(m).actionset_factory, b)

POMDPs.reward(m::KBanditFundingProblem{A, B}, s::CausalState, a::A) where {A, B} = expectedutility(rewardmodel(m), s, a)

POMDPs.initialstate(m::KBanditFundingMDP) = POMDPTools.Deterministic(mdp(m).curr_state) 

# GenerativeBeliefMDP uses this method to figure out the belief type of the POMDP so I can't just return the causal state.
POMDPs.initialstate(m::KBanditFundingPOMDP) = initialbelief(m)

function POMDPs.observation(pomdp::KBanditFundingPOMDP{A, B}, s::CausalState, a::A, sp::CausalState) where {A, B}
    return MultiStudySampleDistribution(Dict(pid => StudySampleDistribution(getprogramstate(s, pid), pomdp.mdp.studysamplesize) for pid in get_evaluated_program_ids(a)))
end

function POMDPs.observation(pomdp::ProgramBanditWrapper{A, B}, s::ProgramCausalState, a::A, sp::ProgramCausalState) where {A, B}
    @assert evaluates(a, getprogramid(s))

    return StudySampleDistribution(s, pomdp.wrapped_problem.mdp.studysamplesize)
end
