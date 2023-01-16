
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

struct RegularizedHyperparam <: AbstractHyperparam
    mu_sd::Float64 
    tau_mean::Float64
    tau_sd::Float64 
    sigma_sd::Float64 
    #eta_sd::Vector{Float64}
    eta_mu_sd::Float64
    eta_tau_sd::Float64

    RegularizedHyperparam(; mu_sd, tau_mean, tau_sd, sigma_sd, eta_mu_sd, eta_tau_sd) = new(mu_sd, tau_mean, tau_sd, sigma_sd, eta_mu_sd, eta_tau_sd)
end

function Base.rand(rng::Random.AbstractRNG, h::RegularizedHyperparam)
    μ = Base.rand(rng, Distributions.Normal(0, h.mu_sd))
    τ = Base.rand(rng, Distributions.Normal(h.tau_mean, h.tau_sd))
    σ = Base.rand(rng, truncated(Distributions.Normal(0, h.sigma_sd), 0, Inf))
    η_μ = Base.rand(rng, truncated(Distributions.Normal(0, h.eta_mu_sd), 0, Inf))
    η_τ = Base.rand(rng, truncated(Distributions.Normal(0, h.eta_tau_sd), 0, Inf))

    return μ, τ, σ, η_μ, η_τ
end

struct InvGammaHyperparam <: AbstractHyperparam
    mu_sd::Float64 
    tau_mean::Float64
    tau_sd::Float64 
    sigma_alpha::Float64 
    sigma_theta::Float64 
    eta_mu_alpha::Float64
    eta_mu_theta::Float64
    eta_tau_alpha::Float64
    eta_tau_theta::Float64

    function InvGammaHyperparam(; mu_sd, tau_mean, tau_sd, sigma_alpha, sigma_theta, eta_mu_alpha, eta_mu_theta, eta_tau_alpha, eta_tau_theta) 
        return new(mu_sd, tau_mean, tau_sd, sigma_alpha, sigma_theta, eta_mu_alpha, eta_mu_theta, eta_tau_alpha, eta_tau_theta)
    end
end

function Base.rand(rng::Random.AbstractRNG, h::InvGammaHyperparam)
    μ = Base.rand(rng, Distributions.Normal(0, h.mu_sd))
    τ = Base.rand(rng, Distributions.Normal(h.tau_mean, h.tau_sd))
    σ = Base.rand(rng, Distributions.InverseGamma(h.sigma_alpha, h.sigma_theta))
    η_μ = Base.rand(rng, Distributions.InverseGamma(h.eta_mu_alpha, h.eta_mu_theta))
    η_τ = Base.rand(rng, Distributions.InverseGamma(h.eta_tau_alpha, h.eta_tau_theta))

    return μ, τ, σ, η_μ, η_τ
end

@model function sim_model(hyperparam::RegularizedHyperparam, datasets = missing; n_sim_study = 0, n_sim_obs = 0, multilevel = true)
    if datasets === missing 
        n_study = n_sim_study 
        datasets = [StudyDataset(n_sim_obs) for i in 1:n_sim_study]
    else 
        n_study = length(datasets) 
    end

    μ_toplevel ~ Normal(0, hyperparam.mu_sd)
    τ_toplevel ~ Normal(hyperparam.tau_mean, hyperparam.tau_sd)
    σ_toplevel ~ truncated(Normal(0, hyperparam.sigma_sd), 0, Inf)

    η_toplevel = [0.0, 0.0] 

    if multilevel
        η_toplevel ~ arraydist([truncated(Normal(0, hyperparam.eta_mu_sd), 0, Inf), truncated(Normal(0, hyperparam.eta_tau_sd), 0, Inf)])
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
    hyperparam::RegularizedHyperparam
    iter
    chains
    multilevel::Bool

    TuringModel(hyperparam::RegularizedHyperparam; iter = 500, chains = 4, multilevel = true) = new(hyperparam, iter, chains, multilevel)
end

function sample(m::TuringModel, datasets::Vector{StudyDataset})
    @pipe sim_model(m.hyperparam, datasets; multilevel = m.multilevel) |>
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