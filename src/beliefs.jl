Base.iterate(b::AbstractBelief) = iterate(b.progbeliefs)
Base.iterate(b::AbstractBelief, n) = iterate(b.progbeliefs, n)
Base.getindex(b::AbstractBelief, i) = b.progbeliefs[i]
Base.length(b::AbstractBelief) = length(b.progbeliefs)

struct ProgramBelief 
    state_samples::ParticleFilters.AbstractParticleBelief
    last_observed_state_samples::Union{Nothing, ParticleFilters.AbstractParticleBelief}
    pid::Int64
    data::Vector{StudyDataset}
    posterior_summary_stats
end

ProgramBelief(state_samples::ParticleFilters.AbstractParticleBelief, pid::Int64) = ProgramBelief(state_samples, nothing, pid, StudyDataset[], nothing)

function ProgramBelief(m::AbstractBayesianModel, data::Vector{StudyDataset}, pid::Int64)
    ndatasets = length(data) 

    logger = Logging.SimpleLogger(Logging.Error)
    samples = Logging.with_logger(logger) do 
        sample(m, data)
    end

    sample_cols = names(samples)

    η = "η_toplevel[1]" in sample_cols && "η_toplevel[1]" in sample_cols ? (samples[:, "η_toplevel[1]"], samples[:, "η_toplevel[2]"]) : (0.0, 0.0) 

    pdgps = ProgramDGP.(samples.μ_toplevel, samples.τ_toplevel, samples.σ_toplevel, η..., pid)
    state_samples = ParticleFilters.ParticleCollection(ProgramCausalState.(pdgps, samples.μ_predict, samples.τ_predict, samples.σ_toplevel, pid))

    last_observed_state = ParticleFilters.ParticleCollection(ProgramCausalState.(pdgps, samples[:, "μ_study[$ndatasets]"], samples[:, "τ_study[$ndatasets]"], samples.σ_toplevel, pid))

    #post_stats = (mean(samples.μ_toplevel), mean(samples.τ_toplevel), mean(samples.σ_toplevel), η...)
    post_stats = (mean(samples.μ_toplevel), mean(samples.τ_toplevel), mean(samples.σ_toplevel))

    return ProgramBelief(state_samples, last_observed_state, pid, data, post_stats)
end

function ProgramBelief(m::OlsModel, data::Vector{StudyDataset}, pid::Int64)
    logger = Logging.SimpleLogger(Logging.Error)
    samples = Logging.with_logger(logger) do 
        sample(m, data)
    end

    pdgp = ProgramDGP(samples.μ_toplevel[1], samples.τ_toplevel[1], samples.σ_toplevel[1], pid)
    last_observed_state = est_state = ParticleFilters.ParticleCollection([ProgramCausalState(pdgp, samples.μ_toplevel[1], samples.τ_toplevel[1], samples.σ_toplevel[1], pid)])

    post_stats = (samples.μ_toplevel[1], samples.τ_toplevel[1], samples.σ_toplevel[1])

    return ProgramBelief(est_state, last_observed_state, pid, data, post_stats)
end

programid(pb::ProgramBelief) = pb.pid

data(pb::ProgramBelief) = pb.data

function utility_particles(r::AbstractRewardModel, bpb::ProgramBelief, a::Union{AbstractFundingAction, Bool})
    return [expectedutility(r, ParticleFilters.particle(bpb.state_samples, i), a) for i in 1:ParticleFilters.n_particles(bpb.state_samples)]
end

function expectedutility(r::AbstractRewardModel, bpb::ProgramBelief, a::Union{AbstractFundingAction, Bool}) 
    return mean(utility_particles(r, bpb, a))
end

state_samples(bpb::ProgramBelief) = bpb.state_samples
last_state_samples(bpb::ProgramBelief) = bpb.last_observed_state_samples

function Base.show(io::IO, pb::ProgramBelief) 
    if pb.posterior_summary_stats === nothing
        print("ProgramBelief(<particles>)")
    else
        Printf.@printf(
            #io, "ProgramBelief({E[μ] = %.2f, E[τ] = %.2f, E[σ] =  %.2f, E[η] = (%.2f, %.2f))", pb.posterior_summary_stats... 
            io, "ProgramBelief({E[μ] = %.2f, E[τ] = %.2f, E[σ] = %.2f)", pb.posterior_summary_stats... 
        ) 
    end
end

Base.rand(rng::Random.AbstractRNG, pb::ProgramBelief) = Base.rand(rng, pb.state_samples)

struct Belief <: AbstractBelief
    progbeliefs::Vector{ProgramBelief}
end

Belief(datasets::Vector{Vector{StudyDataset}}, m::AbstractLearningModel) = Belief([ProgramBelief(m, datasets[pid], pid) for pid in 1:length(datasets)])

data(b::Belief) = data.(b.progbeliefs)

function Base.rand(rng::Random.AbstractRNG, belief::Belief)
    progstates = [Base.rand(rng, belief.progbeliefs[pid]) for pid in 1:length(belief.progbeliefs)]
    progdgps = dgp.(progstates)

    return CausalState(DGP(progdgps), progstates)
end

expectedutility(r::AbstractRewardModel, b::Belief, a::AbstractFundingAction) = sum(expectedutility(r, pb, a) for pb in b.progbeliefs)

struct FundingUpdater{M <: AbstractLearningModel} <: POMDPs.Updater
    model::M
end

POMDPs.initialize_belief(::FundingUpdater, belief::Belief) = belief

POMDPs.update(updater::FundingUpdater, belief_old::Belief, a::AbstractFundingAction, o::EvalObservation) = Belief([POMDPs.update(updater, pb, a, o) for pb in belief_old.progbeliefs])

function POMDPs.update(updater::FundingUpdater, belief_old::ProgramBelief, a::AbstractFundingAction, o::EvalObservation) 
    if evaluates(a, belief_old) 
        new_data = copy(data(belief_old))

        @pipe programid(belief_old) |>
            programobs(o, _) |>
            getdataset(_) |>
            push!(new_data, _)

        return ProgramBelief(updater.model, new_data, belief_old.pid) 
    else
        return belief_old
    end
end