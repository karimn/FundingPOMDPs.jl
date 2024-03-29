expectedutility(r::AbstractRewardModel, w::Rewardable, a::AbstractFundingAction) = expectedutility(r, w, implements(a, w))

struct ExponentialUtilityModel <: AbstractRewardModel
    α::Float64
end

utility(m::ExponentialUtilityModel, outcome) = 1 - exp(- m.α * outcome)

expectedutility(m::ExponentialUtilityModel, μ::Float64, σ::Float64) = 1 - exp(- m.α * μ + m.α^2 * σ^2 / 2)

struct RiskNeutralUtilityModel <: AbstractRewardModel
end

utility(m::RiskNeutralUtilityModel, outcome) = outcome 

expectedutility(m::RiskNeutralUtilityModel, μ::Float64, σ::Float64) = μ
