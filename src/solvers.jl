
struct GreedyPlanner{M} <: POMDPs.Policy where M <: POMDPs.POMDP
    pomdp::M
    nthbest::Int
end

POMDPs.value(p::GreedyPlanner, b::AbstractBelief, a::AbstractFundingAction) = expectedutility(rewardmodel(p.pomdp), b, a)

POMDPs.action(p::GreedyPlanner, b::AbstractBelief) = sort(POMDPs.actions(p.pomdp); by = a -> POMDPs.value(p, b, a), rev = true)[p.nthbest]

POMDPs.value(p::GreedyPlanner, b::AbstractBelief) = POMDPs.value(p, b, POMDPs.action(p, b))

#POMDPs.updater(p::BayesianGreedyPlanner) = FullBayesianUpdater(hyperparam(p.pomdp))

@with_kw_noshow struct GreedySolver <: POMDPs.Solver
    nthbest::Int = 1
end

POMDPs.solve(sol::GreedySolver, m::POMDPs.POMDP) = GreedyPlanner(m, sol.nthbest) 
