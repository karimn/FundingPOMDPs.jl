struct ProgramDGP <: AbstractProgramDGP 
    μ::Float64
    τ::Float64
    σ::Float64
    η::Tuple{Float64, Float64}

    programid::Int64
end

function ProgramDGP(hyperparam::Hyperparam, rng::Random.AbstractRNG, programid::Int64) 
    μ = Base.rand(rng, Distributions.Normal(0, hyperparam.mu_sd))
    τ = Base.rand(rng, Distributions.Normal(hyperparam.tau_mean, hyperparam.tau_sd))
    σ = Base.rand(rng, truncated(Distributions.Normal(0, hyperparam.sigma_sd), 0, Inf))
    η = [Base.rand(rng, truncated(Distributions.Normal(0, hyperparam.eta_sd[i]), 0, Inf)) for i in 1:2]

    ProgramDGP(μ, τ, σ, Tuple{Float64, Float64}(η), programid) 
end

Base.show(io::IO, pdgp::ProgramDGP) = Printf.@printf(io, "ProgramDGP(μ = %.2f, τ = %.2f, σ =  %.2f, η = [%.2f, %.2f])", pdgp.μ, pdgp.τ, pdgp.σ, pdgp.η[1], pdgp.η[2])  

getprogramid(pd::ProgramDGP) = pd.programid

struct ProgramCausalState <: AbstractProgramState 
    μ::Float64
    τ::Float64
    σ::Float64

    programid::Int64
end

function Base.rand(rng::Random.AbstractRNG, pd::ProgramDGP) 
    μ_study = pd.μ + Base.rand(rng, Distributions.Normal(0, pd.η[1]))
    τ_study = pd.τ + Base.rand(rng, Distributions.Normal(0, pd.η[2]))

    return ProgramCausalState(μ_study, τ_study, pd.σ, pd.programid)
end

getprogramid(ps::ProgramCausalState) = ps.programid

function Base.rand(rng::Random.AbstractRNG, ps::ProgramCausalState, samplesize::Int64 = 50) 
    y_control = Base.rand(rng, Normal(ps.μ, ps.σ), samplesize) 
    y_treated = Base.rand(rng, Normal(ps.τ, ps.σ), samplesize) 

    return ProgramEvalObservation(StudyDataset(y_control, y_treated), getprogramid(ps))
end

function logpdf(ps::ProgramCausalState, ds::StudyDataset) 
    controldist = Distributions.Normal(ps.μ, ps.σ) 
    treateddist = Distributions.Normal(ps.μ + ps.τ, ps.σ) 

    Σcontrollogpdf = reduce(ds.y_control, init = 0) do total, y
        total + log(pdf(controldist, y))
    end

    Σtreatedlogpdf = reduce(ds.y_treated, init = 0) do total, y
        total + log(pdf(treateddist, y))
    end

    return Σcontrollogpdf + Σtreatedlogpdf
end

logpdf(ps::ProgramCausalState, progobs::AbstractProgramEvalObservation) = logpdf(ps, progobs.d)

Distributions.pdf(ps::ProgramCausalState, ds::StudyDataset) = exp(logpdf(ps, ds))
Distributions.pdf(ps::ProgramCausalState, progobs::AbstractProgramEvalObservation) = exp(logpdf(ps, progobs))

Base.show(io::IO, ps::ProgramCausalState) = Printf.@printf(io, "ProgramCausalState(μ = %.2f, τ = %.2f, σ =  %.2f)", ps.μ, ps.τ, ps.σ)  

struct DGP <: AbstractDGP
    programdgps::Dict{Int64, AbstractProgramDGP}
end

 function DGP(hyperparam::Hyperparam, rng::Random.AbstractRNG, numprograms::Int64, ::Type{T} = ProgramDGP) where {T <: AbstractProgramDGP}
    DGP(Dict(i => T(hyperparam, rng, i) for i in 1:numprograms)) 
end

numprograms(dgp::DGP) = length(dgp.programdgps)

struct CausalState <: AbstractState 
    programstates::Dict{Int64, AbstractProgramState}
end

Base.rand(rng::Random.AbstractRNG, dgp::DGP) = CausalState(Dict(pid => Base.rand(rng, pdgp) for (pid, pdgp) in dgp.programdgps))

Base.rand(rng::Random.AbstractRNG, s::CausalState, samplesize::Int64 = 50) = EvalObservation([Base.rand(rng, ps, samplesize) for ps in s.programstates])

numprograms(s::CausalState) = length(s.programstates)

getprogramstate(s::CausalState, id) = s.programstates[id]

struct CausalStateDistribution
    dgp::AbstractDGP
end

Base.rand(rng::Random.AbstractRNG, sd::CausalStateDistribution) = Base.rand(rng, sd.dgp) 
