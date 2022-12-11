struct FullBayesianBelief
    hist::AbstractVector{@NamedTuple{a::AbstractFundingAction, o::EvalObservation}}
    datasets::Vector{Vector{StudyDataset}}
    posterior_samples::Vector{DataFrame}
end

function FullBayesianBelief(datasets::Vector{Vector{StudyDataset}}, hyperparam::Hyperparam)
    samples = map(datasets) do ds
        model = sim_model(hyperparam, ds)
        return @pipe DataFrame(Turing.sample(model, Turing.NUTS(), Turing.MCMCThreads(), 500, 4)) |> 
            select(_, "μ_toplevel", "τ_toplevel", "σ_toplevel", "η_toplevel[1]", "η_toplevel[2]") 
    end
        
    return FullBayesianBelief([], datasets, samples)
end

function FullBayesianBelief(a::AbstractFundingAction, o::EvalObservation, hyperparam::Hyperparam, priorbelief::FullBayesianBelief)
    datasets = copy(priorbelief.datasets) 
    samples = copy(priorbelief.posterior_samples)
    hist = copy(priorbelief.hist)
    append!(hist, (:a => a, :o => o))

    for (pid, ds) in getdatasets(o)
        model = sim_model(hyperparam, ds)

        append!(datasets[pid], ds) 
        samples[pid] = @pipe DataFrame(Turing.sample(model, Turing.NUTS(), Turing.MCMCThreads(), 500, 4)) |> 
            select("μ_toplevel", "τ_toplevel", "σ_toplevel", "η_toplevel[1]", "η_toplevel[2]")    
    end

    return FullBayesianBelief(hist, datasets, samples)
end

POMDPs.history(b::FullBayesianBelief) = b.hist

function POMDPs.rand(rng::Random.AbstractRNG, belief::FullBayesianBelief)
    numprog = length(belief.posterior_samples)
    randprogstates = Vector{ProgramCausalState}(undef, numprog)

    for pid in 1:numprog
        df = belief.posterior_samples[pid]
        randrow = df[StatsBase.sample(rng, axes(df, 1), 1), :]

        randprogstates[pid] = ProgramCausalState(rng, randrow.μ_toplevel[1], randrow.τ_toplevel[1], randrow.σ_toplevel[1], Tuple(randrow[1, ["η_toplevel[1]", "η_toplevel[2]"]]), pid)
    end

    return CausalState(Dict(getprogramid(progstate) => progstate for progstate in randprogstates), nothing)
end

struct FullBayesianUpdater <: POMDPs.Updater
    hyperparam::Hyperparam
end

POMDPs.initialize_belief(::FullBayesianUpdater, belief::FullBayesianBelief) = belief

function POMDPs.update(updater::FullBayesianUpdater, belief_old::FullBayesianBelief, a::AbstractFundingAction, o::EvalObservation)
    return FullBayesianBelief(a, o, updater.hyperparam, belief_old)
end