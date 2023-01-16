
struct StudyDataset
    y_control::Vector{Float64}
    y_treated::Vector{Float64}
end

function StudyDataset(n_control::Int64, n_treated::Int64)
    StudyDataset(Vector{Float64}(undef, n_control), Vector{Float64}(undef, n_treated))
end

function StudyDataset(n::Int64)
    StudyDataset(n, n)
end

function Base.show(io::IO, dataset::StudyDataset) 
    Printf.@printf(io, "StudyDataset(sample means = (%.2f, %.2f), sample SD = (%.2f, %.2f))", mean(dataset.y_control), mean(dataset.y_treated), std(dataset.y_control; corrected = false), std(dataset.y_treated; corrected = false))  
end

const StudyHistory = Vector{StudyDataset}

struct Priors
    μ_prior::Distributions.ContinuousUnivariateDistribution
    τ_prior::Distributions.ContinuousUnivariateDistribution
    σ_prior::Distributions.ContinuousUnivariateDistribution
    η_μ_prior::Distributions.ContinuousUnivariateDistribution
    η_τ_prior::Distributions.ContinuousUnivariateDistribution

    Priors(; μ, τ, σ, η_μ, η_τ) = new(μ, τ, σ, η_μ, η_τ)
end

function Base.rand(rng::Random.AbstractRNG, p::Priors)
    μ = Base.rand(rng, p.μ_prior)
    τ = Base.rand(rng, p.τ_prior)
    σ = Base.rand(rng, p.σ_prior)
    η_μ = Base.rand(rng, p.η_μ_prior) 
    η_τ = Base.rand(rng, p.η_τ_prior) 

    return μ, τ, σ, η_μ, η_τ
end

@model function sim_model(priors::Priors, datasets = missing; n_sim_study = 0, n_sim_obs = 0, multilevel = true)
    if datasets === missing 
        n_study = n_sim_study 
        datasets = [StudyDataset(n_sim_obs) for i in 1:n_sim_study]
    else 
        n_study = length(datasets) 
    end

    μ_toplevel ~ priors.μ_prior 
    τ_toplevel ~ priors.τ_prior 
    σ_toplevel ~ priors.σ_prior 

    η_toplevel = [0.0, 0.0] 

    if multilevel
        η_toplevel ~ arraydist([priors.η_μ_prior, priors.η_τ_prior])
        μ_study ~ filldist(Normal(μ_toplevel, η_toplevel[1]), n_study)
        τ_study ~ filldist(Normal(τ_toplevel, η_toplevel[2]), n_study)
    else
        μ_study = μ_toplevel
        τ_study = τ_toplevel
    end
   
    for ds_index in 1:n_study
        for i in eachindex(datasets[ds_index].y_control) 
            datasets[ds_index].y_control[i] ~ Normal(μ_study[ds_index], σ_toplevel)
        end

        for i in eachindex(datasets[ds_index].y_treated) 
            datasets[ds_index].y_treated[i] ~ Normal(μ_study[ds_index] + τ_study[ds_index], σ_toplevel)
        end
    end

    return (datasets = datasets, μ_toplevel = μ_toplevel, μ_study = μ_study)
end

struct TuringModel <: AbstractBayesianModel 
    priors::Priors
    iter
    chains
    multilevel::Bool

    TuringModel(priors::Priors; iter = 500, chains = 4, multilevel = true) = new(priors, iter, chains, multilevel)
end

function sample(m::TuringModel, datasets::Vector{StudyDataset})
    @pipe sim_model(m.priors, datasets; multilevel = m.multilevel) |>
        Turing.sample(_, Turing.NUTS(), Turing.MCMCThreads(), m.iter, m.chains) |> 
        DataFrame(_) |>
        select(_, :μ_toplevel, :τ_toplevel, :σ_toplevel, r"η_toplevel", r"μ_study", r"τ_study") 
end 

#=
struct StanModel <: AbstractBayesianModel
    hyperparam::Hypergeometric
    model::StanSample.StampleModel
end

function StanModel(hyperparam::Hypergeometric)
    StanModel(hyperparam, StanSample.SampleModel("funding", "../stan/sim_model.stan", num_threads = 4, ))
end
=#