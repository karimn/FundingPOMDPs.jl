struct CausalStateParticleBelief <: AbstractBelief
    progbeliefs::Vector{ProgramBelief}
    bayesian_origin::Union{Nothing, FullBayesianBelief}

    function CausalStateParticleBelief(progbeliefs::Vector{ProgramBelief}, origin::Union{Nothing, FullBayesianBelief} = nothing) 
        return new(progbeliefs, origin)
    end
end

expectedutility(m::ExponentialUtilityModel, b::CausalStateParticleBelief, a::AbstractFundingAction) = sum(expectedutility(m, pb, a) for pb in b.progbeliefs)

function Base.convert(::Type{DataFrames.DataFrame}, cspb::CausalStateParticleBelief)
    dfs = map(cspb.progbeliefs) do pb
        prog_state_samples = state_samples(pb) 
        npart = ParticleFilters.n_particles(prog_state_samples)
        df = DataFrames.DataFrame(w = Vector{Float64}(undef, npart), μ = Vector{Float64}(undef, npart), τ = Vector{Float64}(undef, npart), σ = Vector{Float64}(undef, npart), pid = Vector{Int}(undef, npart))

        for i in 1:npart
            pcs = ParticleFilters.particle(prog_state_samples, i) 
            df[i, :] = (ParticleFilters.weight(pb, i), pcs.μ, pcs.τ, pcs.σ, getprogramid(pcs))
        end

        return df
    end

    return hcat(dfs..., makeunique = true)
end

Base.rand(rng::Random.AbstractRNG, b::CausalStateParticleBelief) = CausalState([Base.rand(rng, pb) for pb in b.progbeliefs])

function POMDPs.support(b::CausalStateParticleBelief)
    progsupports = [POMDPs.support(pb) for pb in b.progbeliefs]

    return [CausalState([progstatestuple...]) for progstatestuple in zip(progsupports...)]
end

original_bayesian_beliefs(b::CausalStateParticleBelief) = b.bayesian_origin

struct MultiBootstrapFilter <: POMDPs.Updater    
    filters::Vector{ParticleFilters.BasicParticleFilter}
    bayes_updater::FullBayesianUpdater 
end

function MultiBootstrapFilter(model::KBanditFundingPOMDP, n::Int, fbu::FullBayesianUpdater, rng::Random.AbstractRNG = Random.GLOBAL_RNG)  
    return MultiBootstrapFilter(
        [ParticleFilters.BasicParticleFilter(pbandit, pbandit, ParticleFilters.LowVarianceResampler(n), n, rng) for pbandit in programbandits(model)],
        fbu
    )
end

function POMDPs.update(updater::MultiBootstrapFilter, belief_old::CausalStateParticleBelief, a::AbstractFundingAction, o::EvalObservation)
    new_progbeliefs = copy(belief_old.progbeliefs)

    for (pid, po) in o.programobs
        new_progbeliefs[pid] = ProgramBelief(POMDPs.update(updater.filters[pid], state_samples(new_progbeliefs[pid]), a, po), pid)
    end

    return CausalStateParticleBelief(new_progbeliefs)
end

"""
    POMDPTools.update_info(updater::MultiBootstrapFilter, belief_old::CausalStateParticleBelief, a::AbstractFundingAction, o::EvalObservation)

    I'm using this method to capture updating when making actual step updates, not planning updates. Step updates mean we get a new dataset from the 
    _actual_ state and update our Bayesian beliefs. The planner would directly call the update() methods while the simulation code would call
    update_info() for every step.
"""
function POMDPTools.update_info(updater::MultiBootstrapFilter, belief_old::CausalStateParticleBelief, a::AbstractFundingAction, o::EvalObservation)
    updated_bayes_b = POMDPs.update(updater.bayes_updater, original_bayesian_beliefs(belief_old), a, o)
    updated_particle_b = POMDPs.initialize_belief(updater, updated_bayes_b)

    return updated_particle_b, nothing
end

POMDPs.initialize_belief(::MultiBootstrapFilter, belief::CausalStateParticleBelief) = belief

POMDPs.initialize_belief(updater::MultiBootstrapFilter, belief::FullBayesianBelief) = CausalStateParticleBelief(belief.progbeliefs, belief)

