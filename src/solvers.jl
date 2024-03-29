
struct GreedyPlanner{A} <: POMDPs.Policy where A <: AbstractFundingAction 
    pomdp::FundingPOMDPs.POMDP{A}
    nthbest::Int
end

POMDPs.value(p::GreedyPlanner, b::AbstractBelief, a::AbstractFundingAction) = expectedutility(rewardmodel(p.pomdp), b, a)

POMDPs.action(p::GreedyPlanner, b::AbstractBelief) = sort(POMDPs.actions(p.pomdp, b); by = a -> POMDPs.value(p, b, a), rev = true)[p.nthbest]

POMDPs.action(p::GreedyPlanner{SeparateImplementEvalAction}, b::AbstractBelief) = sort(POMDPs.actions(p.pomdp, b); by = a -> POMDPs.value(p, b, ImplementOnlyAction(get_evaluated_program_ids(a))), rev = true)[p.nthbest]

POMDPs.value(p::GreedyPlanner, b::AbstractBelief) = POMDPs.value(p, b, POMDPs.action(p, b))

#POMDPs.updater(p::BayesianGreedyPlanner) = FullBayesianUpdater(hyperparam(p.pomdp))

@with_kw_noshow struct GreedySolver <: POMDPs.Solver
    nthbest::Int = 1
end

POMDPs.solve(sol::GreedySolver, m::FundingPOMDPs.POMDP) = GreedyPlanner(m, sol.nthbest) 