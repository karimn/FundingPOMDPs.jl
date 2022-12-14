struct CausalStateParticleBelief <: AbstractBelief
    progbeliefs::Vector{ParticleFilters.AbstractParticleBelief}
end

function expectedutility(m::ExponentialUtilityModel, b::ParticleFilters.AbstractParticleBelief, a::AbstractFundingAction)
    return mean(expectedutility(m, ParticleFilters.particle(b, i), a) for i in ParticleFilters.n_particles(b))
end

function expectedutility(m::ExponentialUtilityModel, b::CausalStateParticleBelief, a::AbstractFundingAction)
    return sum(expectedutility(m, pb, a) for pb in b.progbeliefs)
end

struct MultiBootstrapFilter <: POMDPs.Updater    
    filters::Vector{ParticleFilters.BasicParticleFilter} 
end

function MultiBootstrapFilter(model::KBanditFundingPOMDP, n::Int, rng::Random.AbstractRNG = Random.GLOBAL_RNG)  
    #MultiBootstrapFilter([ParticleFilters.BootstrapFilter(model, n, rng) for i in 1:numprograms(model)])
    MultiBootstrapFilter([ParticleFilters.BasicParticleFilter(pbandit, pbandit, ParticleFilters.LowVarianceResampler(n), n, rng) for pbandit in programbandits(model)])
end

function POMDPs.update(updater::MultiBootstrapFilter, belief_old::CausalStateParticleBelief, a::AbstractFundingAction, o::EvalObservation)
    new_progbeliefs = copy(belief_old.progbeliefs)

    for (pid, po) in o.programobs
        new_progbeliefs[pid] = POMDPs.update(updater.filters[pid], new_progbeliefs[pid], a, po)
    end

    return CausalStateParticleBelief(new_progbeliefs)
end

POMDPs.initialize_belief(::MultiBootstrapFilter, belief::CausalStateParticleBelief) = belief

function POMDPs.initialize_belief(updater::MultiBootstrapFilter, belief::FullBayesianBelief)
     partbs = map(updater.filters, belief.progbeliefs) do pf, pb
        POMDPs.initialize_belief(pf, pb)
    end 

    return CausalStateParticleBelief(partbs)
end 