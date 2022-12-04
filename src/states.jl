
abstract type AbstractState end
abstract type AbstractProgramState end

struct ProgramState <: AbstractProgramState 
    μ::Float64
    τ::Float64
    σ::Float64
    η::Tuple{Float64, Float64}

    programid::Int64
end

getprogramid(ps::ProgramState) = ps.programid

function ProgramState(hyperparam::SimDGP.Hyperparam, rng::Random.AbstractRNG, programid::Int64) 
    μ = Base.rand(rng, Distributions.Normal(0, hyperparam.mu_sd))
    τ = Base.rand(rng, Distributions.Normal(hyperparam.tau_mean, hyperparam.tau_sd))
    σ = Base.rand(rng, truncated(Distributions.Normal(0, hyperparam.sigma_sd), 0, Inf))
    η = [Base.rand(rng, truncated(Distributions.Normal(0, hyperparam.eta_sd[i]), 0, Inf)) for i in 1:2]

    ProgramState(μ, τ, σ, Tuple{Float64, Float64}(η), programid) 
end

function rand(rng::Random.AbstractRNG, ps::ProgramState, samplesize::Int64 = 50) 
    μ_study = ps.μ + rand(rng, Distributions.Normal(0, ps.η[1]))
    τ_study = ps.τ + rand(rng, Distributions.Normal(0, ps.η[2]))

    y_control = rand(rng, Normal(μ_study, ps.σ), ss) 
    y_treated = rand(rng, Normal(τ_study, ps.σ), ss) 

    sd = StudyDataset(y_control, y_treated)

    return ProgramEvalObservation(sd, getprogramid(ps))
end

function logpdf(ps::ProgramState, ds::StudyDataset) 
    controldist = Distributions.Normal(ps.μ, sqrt(ps.σ^2 + ps.η[1]^2)) 
    treateddist = Distributions.Normal(ps.μ + ps.τ, sqrt(ps.σ^2 + ps.η[2]^2)) 

    Σcontrollogpdf = reduce(ds.y_control, init = 0) do total, y
        total + logpdf(controldist, y)
    end

    Σtreatedlogpdf = reduce(ds.y_treated, init = 0) do total, y
        total + logpdf(treateddist, y)
    end

    return Σcontrollogpdf + Σtreatedlogpdf
end

pdf(ps::ProgramState, ds::StudyDataset) = exp(logpdf(ps, ds))

struct State{T <: AbstractProgramState} <: AbstractState 
    programstates::Dict{Int64, T}
end

function State{T}(hyperparam::SimDGP.Hyperparam, rng::Random.AbstractRNG, numprograms::Int64) where {T <: AbstractProgramState}
    State{T}(Dict(i => T(hyperparam, rng, i) for i in 1:numprograms)) 
end

numprograms(s::State{T}) where {T <: AbstractProgramState} = length(s.programstates)