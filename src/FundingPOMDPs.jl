module FundingPOMDPs

import Combinatorics
import Random
import Printf
import DynamicPPL
import StatsBase
import ParticleFilters
import Logging

using Statistics

using POMDPTools
using POMDPs
using POMDPLinter
using Distributions
using Parameters
using Turing, GLM
using DataFrames, DataFramesMeta
using Pipe

abstract type AbstractRewardModel end
abstract type Rewardable end

abstract type AbstractBelief <: Rewardable end
abstract type AbstractLearningModel end
abstract type AbstractBayesianModel <: AbstractLearningModel end
abstract type AbstractFrequentistModel <: AbstractLearningModel end

abstract type AbstractDGP <: Rewardable end
abstract type AbstractProgramDGP <: Rewardable end
abstract type AbstractState <: Rewardable end
abstract type AbstractProgramState <: Rewardable end

abstract type AbstractStudySampleDistribution end
abstract type AbstractSampleDistribution end
abstract type AbstractEvalObservation end
abstract type AbstractProgramEvalObservation end

abstract type AbstractFundingAction end
abstract type AbstractActionSet end
abstract type AbstractActionSetFactory{A} end

include("SimDGP.jl")
include("reward.jl")
include("dgp.jl")
include("states.jl")
include("observations.jl")

abstract type MDP{A <: AbstractFundingAction} <: POMDPs.MDP{CausalState, A} end
abstract type POMDP{A <: AbstractFundingAction} <: POMDPs.POMDP{CausalState, A, EvalObservation} end

include("beliefs.jl")
include("actions.jl")
include("actionsets.jl")
include("kbandit.jl")
include("solvers.jl")
include("particles.jl")
include("util.jl")

export Rewardable
export StudyDataset, Priors, sim_model 
export AbstractDGP, AbstractProgramDGP, AbstractState, expectedlevel
export DGP, ProgramDGP, CausalState, dgp, next_state, prev_state
export AbstractEvalObservation
export EvalObservation, getdatasets
export AbstractRewardModel 
export ExponentialUtilityModel, RiskNeutralUtilityModel, expectedutility, utility
export AbstractFundingAction 
export ImplementEvalAction, ImplementOnlyAction, SeparateImplementEvalAction, implements, evaluates
export AbstractActionSet, AbstractFundingAction, AbstractActionSetFactory
export actions
export KBanditFundingMDP, KBanditFundingPOMDP
export KBanditActionSet, SelectProgramSubsetActionSetFactory, SeparateImplementAndEvalActionSetFactory, ExploreOnlyActionSetFactory
export initialbelief, rewardmodel, initialstate
export Belief, ProgramBelief, FundingUpdater
export utility_particles, programid, state_samples, last_state_samples, programbeliefs, data
export GreedySolver
export MultiBootstrapFilter, CausalStateParticleBelief
export TuringModel, OlsModel, sample
export get_rewards_data, get_beliefs_data, get_dgp_data, get_states_data, get_actions_data

end # module