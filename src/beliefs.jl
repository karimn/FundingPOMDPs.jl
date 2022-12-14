struct FullBayesianProgramBelief
    posterior_samples::DataFrame
    pid::Int64
end

function expectedutility(r::ExponentialUtilityModel, pb::FullBayesianProgramBelief, impl::Bool)
    return mean(
        expectedutility(
            r, 
            impl ? pb.posterior_samples.μ_toplevel + pb.posterior_samples.τ_toplevel : pb.posterior_samples.μ_toplevel, 
            sqrt.(pb.posterior_samples.σ_toplevel.^2 + pb.posterior_samples[:, "η_toplevel[1]"].^2 .+ (impl ? pb.posterior_samples[:, "η_toplevel[2]"].^2 : 0.0))
        )
    )
end

expectedutility(r::ExponentialUtilityModel, pb::FullBayesianProgramBelief, a::AbstractFundingAction) = expectedutility(r, pb, implements(a, pb.pid))

function Base.show(io::IO, pb::FullBayesianProgramBelief) 
    Printf.@printf(
        io, "FullBayesianProgramBelief({E[μ] = %.2f, E[τ] = %.2f, E[σ] =  %.2f, E[η] = (%.2f, %.2f))", 
        mean(pb.posterior_samples.μ_toplevel), mean(pb.posterior_samples.τ_toplevel), mean(pb.posterior_samples.σ_toplevel), mean(pb.posterior_samples[:, "η_toplevel[1]"]), mean(pb.posterior_samples[:, "η_toplevel[2]"])
    ) 
end

function Base.rand(rng::Random.AbstractRNG, pb::FullBayesianProgramBelief)
    randrow = pb.posterior_samples[StatsBase.sample(rng, axes(pb.posterior_samples, 1), 1), :]
    pdgp = ProgramDGP(randrow.μ_toplevel[1], randrow.τ_toplevel[1], randrow.σ_toplevel[1], Tuple(randrow[1, ["η_toplevel[1]", "η_toplevel[2]"]]), pb.pid)

    #return ProgramCausalState(rng, randrow.μ_toplevel[1], randrow.τ_toplevel[1], randrow.σ_toplevel[1], Tuple(randrow[1, ["η_toplevel[1]", "η_toplevel[2]"]]), pb.pid)
    return Base.rand(rng, pdgp) #ProgramCausalState(rng, pdgp, pb.pid)
end

struct FullBayesianBelief{M <: AbstractBayesianModel} <: AbstractBelief
    datasets::Vector{Vector{StudyDataset}}
    progbeliefs::Vector{FullBayesianProgramBelief}
end

function FullBayesianBelief{M}(datasets::Vector{Vector{StudyDataset}}, hyperparam::Hyperparam) where {M <: AbstractBayesianModel}
    m = M(hyperparam)
    samples = [FullBayesianProgramBelief(sample(m, datasets[pid]), pid) for pid in 1:length(datasets)]
        
    return FullBayesianBelief{M}(datasets, samples)
end

function FullBayesianBelief{M}(a::AbstractFundingAction, o::EvalObservation, hyperparam::Hyperparam, priorbelief::FullBayesianBelief) where {M}
    datasets = deepcopy(priorbelief.datasets) 
    progbeliefs = copy(priorbelief.progbeliefs)
    m = M(hyperparam)

    for (pid, ds) in getdatasets(o)
        push!(datasets[pid], ds) 
        progbeliefs[pid] = FullBayesianProgramBelief(sample(m, datasets[pid]), pid) 
    end

    return FullBayesianBelief{M}(datasets, progbeliefs)
end

function POMDPs.rand(rng::Random.AbstractRNG, belief::FullBayesianBelief)
    progstates = Dict(pid => Base.rand(rng, belief.progbeliefs[pid]) for pid in 1:length(belief.progbeliefs))
    progdgps = Dict(pid => dgp(ps) for (pid, ps) in progstates)

    return CausalState(DGP(progdgps), progstates, nothing)
end

expectedutility(r::ExponentialUtilityModel, b::FullBayesianBelief, a::AbstractFundingAction) = sum(expectedutility(r, pb, a) for pb in b.progbeliefs)

struct FullBayesianUpdater <: POMDPs.Updater
    hyperparam::Hyperparam
end

POMDPs.initialize_belief(::FullBayesianUpdater, belief::FullBayesianBelief) = belief

function POMDPs.update(updater::FullBayesianUpdater, belief_old::FullBayesianBelief{M}, a::AbstractFundingAction, o::EvalObservation) where {M}
    return FullBayesianBelief{M}(a, o, updater.hyperparam, belief_old)
end
