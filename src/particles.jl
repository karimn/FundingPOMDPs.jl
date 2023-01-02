struct CausalStateParticleBelief <: AbstractBelief
    progbeliefs::Vector{ParticleFilters.AbstractParticleBelief}
    bayesian_origin::Union{Nothing, FullBayesianBelief}

    function CausalStateParticleBelief(progbeliefs::Vector{PB}, origin::Union{Nothing, FullBayesianBelief} = nothing) where PB <: ParticleFilters.AbstractParticleBelief 
        return new(progbeliefs, origin)
    end
end


expectedutility(m::ExponentialUtilityModel, b::ParticleFilters.AbstractParticleBelief, a::AbstractFundingAction) = mean(expectedutility(m, ParticleFilters.particle(b, i), a) for i in ParticleFilters.n_particles(b))

expectedutility(m::ExponentialUtilityModel, b::CausalStateParticleBelief, a::AbstractFundingAction) = sum(expectedutility(m, pb, a) for pb in b.progbeliefs)

function Base.convert(::Type{DataFrames.DataFrame}, cspb::CausalStateParticleBelief)
    dfs = map(cspb.progbeliefs) do pb
        npart = ParticleFilters.n_particles(pb)
        df = DataFrames.DataFrame(w = Vector{Float64}(undef, npart), μ = Vector{Float64}(undef, npart), τ = Vector{Float64}(undef, npart), σ = Vector{Float64}(undef, npart), pid = Vector{Int}(undef, npart))

        for i in 1:npart
            pcs = ParticleFilters.particle(pb, i) 
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
        new_progbeliefs[pid] = POMDPs.update(updater.filters[pid], new_progbeliefs[pid], a, po)
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

function POMDPs.initialize_belief(updater::MultiBootstrapFilter, belief::FullBayesianBelief)
#    test = POMDPs.initialize_belief(updater.filters[1], belief.progbeliefs[1])

     #=partbs = map(updater.filters, belief.progbeliefs) do pf, pb
        POMDPs.initialize_belief(pf, pb)
    end=# 

    partbs = [ParticleFilters.ParticleCollection(state_samples(pb)) for pb in belief.progbeliefs]

    return CausalStateParticleBelief(partbs, belief)
end 

