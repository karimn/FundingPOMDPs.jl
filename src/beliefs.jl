struct FullBayesianProgramBelief
    posterior_samples::DataFrame
    pid::Int64
end

function expectedutility(r::ExponentialUtilityModel, pb::FullBayesianProgramBelief, a::AbstractFundingAction)
    return mean(
        expectedutility(
            r, 
            implements(a, pb.pid) ? pb.posterior_samples.μ_toplevel + pb.posterior_samples.τ_toplevel : pb.posterior_samples.μ_toplevel, 
            sqrt.(pb.posterior_samples.σ_toplevel.^2 + pb.posterior_samples[:, "η_toplevel[1]"].^2 .+ (implements(a, pb.pid) ? pb.posterior_samples[:, "η_toplevel[2]"].^2 : 0.0))
        )
    )
end

function Base.show(io::IO, pb::FullBayesianProgramBelief) 
    Printf.@printf(
        io, "FullBayesianProgramBelief({E[μ] = %.2f, E[τ] = %.2f, E[σ] =  %.2f, E[η] = (%.2f, %.2f))", 
        mean(pb.posterior_samples.μ_toplevel), mean(pb.posterior_samples.τ_toplevel), mean(pb.posterior_samples.σ_toplevel), mean(pb.posterior_samples[:, "η_toplevel[1]"]), mean(pb.posterior_samples[:, "η_toplevel[2]"])
    ) 
end

function Base.rand(rng::Random.AbstractRNG, pb::FullBayesianProgramBelief)
    randrow = pb.posterior_samples[StatsBase.sample(rng, axes(df, 1), 1), :]

    return ProgramCausalState(rng, randrow.μ_toplevel[1], randrow.τ_toplevel[1], randrow.σ_toplevel[1], Tuple(randrow[1, ["η_toplevel[1]", "η_toplevel[2]"]]), pb.pid)
end

struct FullBayesianBelief
    datasets::Vector{Vector{StudyDataset}}
    progbeliefs::Vector{FullBayesianProgramBelief}
end

function FullBayesianBelief(datasets::Vector{Vector{StudyDataset}}, hyperparam::Hyperparam)
    samples = Vector{FullBayesianProgramBelief}(undef, length(datasets))

    for pid in 1:length(samples)
        model = sim_model(hyperparam, datasets[pid])
        samples[pid] = @pipe DataFrame(Turing.sample(model, Turing.NUTS(), Turing.MCMCThreads(), 500, 4)) |> 
            select(_, "μ_toplevel", "τ_toplevel", "σ_toplevel", "η_toplevel[1]", "η_toplevel[2]")  |>
            FullBayesianProgramBelief(_, pid)
    end
        
    return FullBayesianBelief(datasets, samples)
end

function FullBayesianBelief(a::AbstractFundingAction, o::EvalObservation, hyperparam::Hyperparam, priorbelief::FullBayesianBelief)
    datasets = deepcopy(priorbelief.datasets) 
    progbeliefs = copy(priorbelief.progbeliefs)

    for (pid, ds) in getdatasets(o)
        push!(datasets[pid], ds) 
        model = sim_model(hyperparam, datasets[pid])
        progbeliefs[pid] = @pipe DataFrame(Turing.sample(model, Turing.NUTS(), Turing.MCMCThreads(), 500, 4)) |> 
            select(_, "μ_toplevel", "τ_toplevel", "σ_toplevel", "η_toplevel[1]", "η_toplevel[2]") |>
            FullBayesianProgramBelief(_, pid)
    end

    return FullBayesianBelief(datasets, progbeliefs)
end

function POMDPs.rand(rng::Random.AbstractRNG, belief::FullBayesianBelief)
    return CausalState(Dict(pid => Base.rand(rng, belief.progbeliefs[pid]) for pid in 1:length(belief.progbeliefs)), nothing)
end

expectedutility(r::ExponentialUtilityModel, b::FullBayesianBelief, a::AbstractFundingAction) = sum(expectedutility(r, pb, a) for pb in b.progbeliefs)

struct FullBayesianUpdater <: POMDPs.Updater
    hyperparam::Hyperparam
end

POMDPs.initialize_belief(::FullBayesianUpdater, belief::FullBayesianBelief) = belief

function POMDPs.update(updater::FullBayesianUpdater, belief_old::FullBayesianBelief, a::AbstractFundingAction, o::EvalObservation)
    return FullBayesianBelief(a, o, updater.hyperparam, belief_old)
end