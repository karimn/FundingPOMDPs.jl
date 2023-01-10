Base.iterate(b::AbstractBelief) = iterate(b.progbeliefs)
Base.iterate(b::AbstractBelief, n) = iterate(b.progbeliefs, n)
Base.getindex(b::AbstractBelief, i) = b.progbeliefs[i]
Base.length(b::AbstractBelief) = length(b.progbeliefs)

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

programid(pb::ProgramBelief) = pb.pid

data(pb::ProgramBelief) = pb.data

function utility_particles(r::ExponentialUtilityModel, bpb::ProgramBelief, a::Union{AbstractFundingAction, Bool})
    return [expectedutility(r, ParticleFilters.particle(bpb.state_samples, i), a) for i in 1:ParticleFilters.n_particles(bpb.state_samples)]
end

function expectedutility(r::ExponentialUtilityModel, bpb::ProgramBelief, a::Union{AbstractFundingAction, Bool}) 
    return mean(utility_particles(r, bpb, a))
end

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

struct FullBayesianBelief <: AbstractBelief
    progbeliefs::Vector{ProgramBelief}
end

function FullBayesianBelief(datasets::Vector{Vector{StudyDataset}}, m::AbstractBayesianModel, rng::Random.AbstractRNG) 
    samples = [ProgramBelief(m, datasets[pid], pid, rng) for pid in 1:length(datasets)]
        
    return FullBayesianBelief(samples)
end

function Base.rand(rng::Random.AbstractRNG, belief::FullBayesianBelief)
    progstates = [Base.rand(rng, belief.progbeliefs[pid]) for pid in 1:length(belief.progbeliefs)]
    progdgps = dgp.(progstates)

    return CausalState(DGP(progdgps), progstates)
end

expectedutility(r::ExponentialUtilityModel, b::FullBayesianBelief, a::AbstractFundingAction) = sum(expectedutility(r, pb, a) for pb in b.progbeliefs)

struct FullBayesianUpdater{M <: AbstractBayesianModel} <: POMDPs.Updater
    rng::Random.AbstractRNG
    bayesian_model::M
end

POMDPs.initialize_belief(::FullBayesianUpdater, belief::FullBayesianBelief) = belief

POMDPs.update(updater::FullBayesianUpdater, belief_old::FullBayesianBelief, a::AbstractFundingAction, o::EvalObservation) = FullBayesianBelief([POMDPs.update(updater, pb, a, o) for pb in belief_old.progbeliefs])

function POMDPs.update(updater::FullBayesianUpdater, belief_old::ProgramBelief, a::AbstractFundingAction, o::EvalObservation) 
    if evaluates(a, belief_old) 
        new_data = copy(data(belief_old))

        @pipe programid(belief_old) |>
            programobs(o, _) |>
            getdataset(_) |>
            push!(new_data, _)

        return ProgramBelief(updater.bayesian_model, new_data, belief_old.pid, updater.rng) 
    else
        return belief_old
    end
end