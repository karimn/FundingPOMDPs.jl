struct CausalStateParticleBelief <: AbstractBelief
    progbeliefs::Vector{ParticleFilters.AbstractParticleBelief}
end

function expectedutility(m::ExponentialUtilityModel, b::ParticleFilters.AbstractParticleBelief, a::AbstractFundingAction)
    return mean(expectedutility(m, ParticleFilters.particle(b, i), a) for i in ParticleFilters.n_particles(b))
end

function expectedutility(m::ExponentialUtilityModel, b::CausalStateParticleBelief, a::AbstractFundingAction)
    return sum(expectedutility(m, pb, a) for pb in b.progbeliefs)
end

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

POMDPs.rand(rng::Random.AbstractRNG, b::CausalStateParticleBelief) = CausalState([POMDPs.rand(rng, pb) for pb in b.progbeliefs])

function POMDPs.support(b::CausalStateParticleBelief)
    progsupports = [POMDPs.support(pb) for pb in b.progbeliefs]

    return [CausalState([progstatestuple...]) for progstatestuple in zip(progsupports...)]
end

struct MultiBootstrapFilter <: POMDPs.Updater    
    filters::Vector{ParticleFilters.BasicParticleFilter} 
end

function MultiBootstrapFilter(model::KBanditFundingPOMDP, n::Int, rng::Random.AbstractRNG = Random.GLOBAL_RNG)  
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