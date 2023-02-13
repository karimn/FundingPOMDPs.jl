struct ProgramDGP <: AbstractProgramDGP 
    μ::Float64
    τ::Float64
    σ::Float64
    η_μ::Float64
    η_τ::Float64

    programid::Int64
end

ProgramDGP(μ::Float64, τ::Float64, σ::Float64, pid::Int64) = ProgramDGP(μ, τ, σ, 0.0, 0.0, pid)

function ProgramDGP(priors::Priors, rng::Random.AbstractRNG, programid::Int64) 
    μ, τ, σ, η_μ, η_τ = Base.rand(rng, priors)

    ProgramDGP(μ, τ, σ, η_μ, η_τ, programid) 
end

Base.show(io::IO, pdgp::ProgramDGP) = Printf.@printf(io, "ProgramDGP(μ = %.2f, τ = %.2f, σ =  %.2f, η = [%.2f, %.2f])", pdgp.μ, pdgp.τ, pdgp.σ, pdgp.η_μ, pdgp.η_τ)  

getprogramid(pd::ProgramDGP) = pd.programid

function expectedutility(r::AbstractRewardModel, progdgp::ProgramDGP, impl::Bool) 
    return expectedutility(
        r, 
        progdgp.μ + (impl ? progdgp.τ : 0), 
        sqrt(progdgp.σ^2 + progdgp.η_μ^2 + (impl ? progdgp.η_τ^2 : 0))
    )
end

struct DGP <: AbstractDGP
    programdgps::Vector{ProgramDGP}
end

function DGP(priors::Priors, rng::Random.AbstractRNG, numprograms::Int64)
    return DGP([ProgramDGP(priors, rng, i) for i in 1:numprograms]) 
end

numprograms(dgp::DGP) = length(dgp.programdgps)

expectedutility(r::AbstractRewardModel, dgp::DGP, a::AbstractFundingAction) = sum(expectedutility(r, pdgp, a) for pdgp in dgp.programdgps)