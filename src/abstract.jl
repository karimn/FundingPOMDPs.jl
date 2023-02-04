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

