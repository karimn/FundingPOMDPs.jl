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

function expectedutility(r::ExponentialUtilityModel, progdgp::ProgramDGP, impl::Bool) 
    return expectedutility(
        r, 
        progdgp.μ + (impl ? progdgp.τ : 0), 
        sqrt(progdgp.σ^2 + progdgp.η[1]^2 + (impl ? progdgp.η[2]^2 : 0))
    )
end

expectedutility(r::ExponentialUtilityModel, progdgp::ProgramDGP, a::AbstractFundingAction) = expectedutility(r, progdgp, implements(a, progdgp.programid))

struct DGP <: AbstractDGP
    programdgps::Dict{Int64, AbstractProgramDGP}
end

 function DGP(hyperparam::Hyperparam, rng::Random.AbstractRNG, numprograms::Int64, ::Type{T} = ProgramDGP) where {T <: AbstractProgramDGP}
    return DGP(Dict(i => T(hyperparam, rng, i) for i in 1:numprograms)) 
end

numprograms(dgp::DGP) = length(dgp.programdgps)

expectedutility(r::ExponentialUtilityModel, dgp::DGP, a::AbstractFundingAction) = sum(expectedutility(r, pdgp, a) for (_, pdgp) in dgp.programdgps)