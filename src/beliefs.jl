struct ProgramBelief 
    state_samples::ParticleFilters.AbstractParticleBelief
    pid::Int64
    data::Vector{StudyDataset}
    posterior_summary_stats

    ProgramBelief(state_samples::ParticleFilters.AbstractParticleBelief, pid::Int64; data::Vector{StudyDataset} = StudyDataset[], post_summary = nothing) = new(state_samples, pid, data, post_summary)
end

function ProgramBelief(m::AbstractBayesianModel, data::Vector{StudyDataset}, pid::Int64, rng::Random.AbstractRNG)
    samples = sample(m, data)

    pdgps = ProgramDGP.(samples.μ_toplevel, samples.τ_toplevel, samples.σ_toplevel, samples[:, "η_toplevel[1]"], samples[:, "η_toplevel[2]"], pid)
    state_samples = ParticleFilters.ParticleCollection(Base.rand.(rng, pdgps))

    post_stats = (mean(samples.μ_toplevel), mean(samples.τ_toplevel), mean(samples.σ_toplevel), mean(samples[:, "η_toplevel[1]"]), mean(samples[:, "η_toplevel[2]"]))

    return ProgramBelief(state_samples, pid; data = data, post_summary = post_stats)
end

data(pb::ProgramBelief) = pb.data

expectedutility(r::ExponentialUtilityModel, bpb::ProgramBelief, a::AbstractFundingAction) = mean(expectedutility(r, ParticleFilters.particle(bpb.state_samples, i), a) for i in ParticleFilters.n_particles(bpb.state_samples))

state_samples(bpb::ProgramBelief) = bpb.state_samples

function Base.show(io::IO, pb::ProgramBelief) 
    if pb.posterior_summary_stats === nothing
        print("ProgramBelief(<particles>)")
    else
        Printf.@printf(
            io, "ProgramBelief({E[μ] = %.2f, E[τ] = %.2f, E[σ] =  %.2f, E[η] = (%.2f, %.2f))", pb.posterior_summary_stats... 
        ) 
    end
end

Base.rand(rng::Random.AbstractRNG, pb::ProgramBelief) = Base.rand(rng, pb.state_samples)

struct FullBayesianBelief{M <: AbstractBayesianModel} <: AbstractBelief
    progbeliefs::Vector{ProgramBelief}
end

function FullBayesianBelief{M}(datasets::Vector{Vector{StudyDataset}}, hyperparam::Hyperparam, rng::Random.AbstractRNG) where {M <: AbstractBayesianModel}
    m = M(hyperparam)
    samples = [ProgramBelief(m, datasets[pid], pid, rng) for pid in 1:length(datasets)]
        
    return FullBayesianBelief{M}(samples)
end

function FullBayesianBelief{M}(a::AbstractFundingAction, o::EvalObservation, hyperparam::Hyperparam, priorbelief::FullBayesianBelief, rng::Random.AbstractRNG) where {M}
    progbeliefs = copy(priorbelief.progbeliefs)
    m = M(hyperparam)

    for (pid, ds) in getdatasets(o)
        new_data = copy(data(progbeliefs[pid]))
        push!(new_data, ds)

        progbeliefs[pid] = ProgramBelief(m, new_data, pid, rng) 
    end

    return FullBayesianBelief{M}(progbeliefs)
end

function Base.rand(rng::Random.AbstractRNG, belief::FullBayesianBelief)
    progstates = [Base.rand(rng, belief.progbeliefs[pid]) for pid in 1:length(belief.progbeliefs)]
    progdgps = dgp.(progstates)

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
