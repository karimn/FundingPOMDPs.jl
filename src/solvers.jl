
struct BayesianGreedyPlanner{M} <: POMDPs.Policy where M <: POMDPs.POMDP
    pomdp::M
end

POMDPs.value(p::BayesianGreedyPlanner, b, a::AbstractFundingAction) = expectedutility(rewardmodel(p.pomdp), b, a)

function POMDPs.action(p::BayesianGreedyPlanner, b)
    best_reward = -Inf
    local best_action

    for a in POMDPs.actions(p.pomdp)
        current_reward = POMDPs.value(p, b, a) 

        if current_reward > best_reward
            best_reward = current_reward
            best_action = a
        end
    end

    return best_action 
end

POMDPs.value(p::BayesianGreedyPlanner, b) = POMDPs.value(p, b, POMDPs.action(p, b))

POMDPs.updater(p::BayesianGreedyPlanner) = FullBayesianUpdater(hyperparam(p.pomdp))

struct BayesianGreedySolver <: POMDPs.Solver
end

POMDPs.solve(sol::BayesianGreedySolver, m::POMDPs.POMDP) = BayesianGreedyPlanner(m) 
