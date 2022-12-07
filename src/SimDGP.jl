using Turing
using DataFrames

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

const StudyHistory = Vector{StudyDataset}

struct Hyperparam
    mu_sd::Float64 
    tau_mean::Float64
    tau_sd::Float64 
    sigma_sd::Float64 
    eta_sd::Vector{Float64}

    Hyperparam(; mu_sd, tau_mean, tau_sd, sigma_sd, eta_sd) = new(mu_sd, tau_mean, tau_sd, sigma_sd, eta_sd)
end

@model function sim_model(hyperparam::Hyperparam, datasets = missing; n_sim_study = 0, n_sim_obs = 0)

    if datasets === missing 
        n_study = n_sim_study 
        datasets = [StudyDataset(n_sim_obs) for i in 1:n_sim_study]
    else 
        n_study = length(datasets) 
    end

    μ_toplevel ~ Normal(0, hyperparam.mu_sd)
    τ_toplevel ~ Normal(hyperparam.tau_mean, hyperparam.tau_sd)
    σ_toplevel ~ truncated(Normal(0, hyperparam.sigma_sd), 0, Inf)
    η_toplevel ~ arraydist([truncated(Normal(0, hyperparam.eta_sd[i]), 0, Inf) for i in 1:2])

    μ_study ~ filldist(Normal(μ_toplevel, η_toplevel[1]), n_study)
    τ_study ~ filldist(Normal(τ_toplevel, η_toplevel[2]), n_study)
   
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
