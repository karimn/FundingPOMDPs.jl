struct FullBayesianProgramBelief
    posterior_samples::DataFrame
    state_samples::Vector{ProgramCausalState}
    pid::Int64

    function FullBayesianProgramBelief(samples::DataFrame, pid::Int64, rng::Random.AbstractRNG)
        pdgps = ProgramDGP.(samples.μ_toplevel, samples.τ_toplevel, samples.σ_toplevel, samples[:, "η_toplevel[1]"], samples[:, "η_toplevel[2]"], pid)
        return new(samples, Base.rand.(rng, pdgps), pid)
    end
end

expectedutility(r::ExponentialUtilityModel, bpb::FullBayesianProgramBelief, a::AbstractFundingAction) = mean(expectedutility(r, s, a) for s in bpb.state_samples)

#=function expectedutility(r::ExponentialUtilityModel, pb::FullBayesianProgramBelief, impl::Bool)
    return mean(
        expectedutility(
            r, 
            impl ? pb.posterior_samples.μ_toplevel + pb.posterior_samples.τ_toplevel : pb.posterior_samples.μ_toplevel, 
            sqrt.(pb.posterior_samples.σ_toplevel.^2 + pb.posterior_samples[:, "η_toplevel[1]"].^2 .+ (impl ? pb.posterior_samples[:, "η_toplevel[2]"].^2 : 0.0))
        )
    )
end
=#

state_samples(bpb::FullBayesianProgramBelief) = bpb.state_samples

function Base.show(io::IO, pb::FullBayesianProgramBelief) 
    Printf.@printf(
        io, "FullBayesianProgramBelief({E[μ] = %.2f, E[τ] = %.2f, E[σ] =  %.2f, E[η] = (%.2f, %.2f))", 
        mean(pb.posterior_samples.μ_toplevel), mean(pb.posterior_samples.τ_toplevel), mean(pb.posterior_samples.σ_toplevel), mean(pb.posterior_samples[:, "η_toplevel[1]"]), mean(pb.posterior_samples[:, "η_toplevel[2]"])
    ) 
end

Base.rand(rng::Random.AbstractRNG, pb::FullBayesianProgramBelief) = Base.rand(rng, pb.state_samples)

struct FullBayesianBelief{M <: AbstractBayesianModel} <: AbstractBelief
    datasets::Vector{Vector{StudyDataset}} # Maybe this should be stored in the program struct?
    progbeliefs::Vector{FullBayesianProgramBelief}
end

function FullBayesianBelief{M}(datasets::Vector{Vector{StudyDataset}}, hyperparam::Hyperparam, rng::Random.AbstractRNG) where {M <: AbstractBayesianModel}
    m = M(hyperparam)
    samples = [FullBayesianProgramBelief(sample(m, datasets[pid]), pid, rng) for pid in 1:length(datasets)]
        
    return FullBayesianBelief{M}(datasets, samples)
end

function FullBayesianBelief{M}(a::AbstractFundingAction, o::EvalObservation, hyperparam::Hyperparam, priorbelief::FullBayesianBelief, rng::Random.AbstractRNG) where {M}
    datasets = deepcopy(priorbelief.datasets) 
    progbeliefs = copy(priorbelief.progbeliefs)
    m = M(hyperparam)

    for (pid, ds) in getdatasets(o)
        push!(datasets[pid], ds) 
        progbeliefs[pid] = FullBayesianProgramBelief(sample(m, datasets[pid]), pid, rng) 
    end

    return FullBayesianBelief{M}(datasets, progbeliefs)
end

function Base.rand(rng::Random.AbstractRNG, belief::FullBayesianBelief)
    progstates = [Base.rand(rng, belief.progbeliefs[pid]) for pid in 1:length(belief.progbeliefs)]
    progdgps = dgp.(progstates) # [dgp(ps) for ps in progstates]

    return CausalState(DGP(progdgps), progstates)
end

expectedutility(r::ExponentialUtilityModel, b::FullBayesianBelief, a::AbstractFundingAction) = sum(expectedutility(r, pb, a) for pb in b.progbeliefs)

struct FullBayesianUpdater <: POMDPs.Updater
    hyperparam::Hyperparam
    rng::Random.AbstractRNG
end

POMDPs.initialize_belief(::FullBayesianUpdater, belief::FullBayesianBelief) = belief

function POMDPs.update(updater::FullBayesianUpdater, belief_old::FullBayesianBelief{M}, a::AbstractFundingAction, o::EvalObservation) where {M}
    return FullBayesianBelief{M}(a, o, updater.hyperparam, belief_old, updater.rng)
end
