abstract type AbstractSampleDistribution end
abstract type AbstractEvalObservation end

struct ProgramEvalObservation
    d::StudyDataset
    programid::Int64
end

getprogramid(o::ProgramEvalObservation) = o.programid

struct EvalObservation <: AbstractEvalObservation
    po::Dict{Int64, ProgramEvalObservation}
end

EvalObservation(pos::Vector{ProgramEvalObservation}) = EvalObservation(Dict(getprogramid(po) => po for po in pos))

struct StudySampleDistribution{T <: AbstractProgramState} <: AbstractSampleDistribution
    programstate::T
    samplesize::Int64
end

getprogramid(d::StudySampleDistribution{T}) where {T <: AbstractProgramState} = getprogramid(d.programstate)
rand(rng::Random.AbstractRNG, d::StudySampleDistribution{T}) where {T <: AbstractProgramState} = rand(rng, d.programstate, d.samplesize) 
logpdf(d::StudySampleDistribution{T}, ds::StudyDataset) where {T <: AbstractProgramState} = logpdf(d.programstate, ds)
pdf(d::StudySampleDistribution{T}, ds::StudyDataset) where {T <: AbstractProgramState} = pdf(d.programstate, ds)
logpdf(d::StudySampleDistribution{T}, o::ProgramEvalObservation) where {T <: AbstractProgramState} = logpdf(d, o.d)
pdf(d::StudySampleDistribution{T}, o::ProgramEvalObservation) where {T <: AbstractProgramState} = pdf(d, o.d)

struct MultiStudySampleDistribution{T <: AbstractProgramState} <: AbstractSampleDistribution
    sds::Dict{Int64, StudySampleDistribution{T}}
end

MultiStudySampleDistribution{T}(sds::Vector{StudySampleDistribution{T}}) where{T <: AbstractProgramState} = MultiStudySampleDistribution{T}(Dict(getprogramid(d) => d for d in sds))
rand(rng::Random.AbstractRNG, msd::MultiStudySampleDistribution{T}) where {T <: AbstractProgramState} = EvalObservation{T}([rand(rng, d) for d in msd.sds])

function logpdf(msd::MultiStudySampleDistribution{T}, o::EvalObservation) where {T <: AbstractProgramState}
    length(msd.sds) == length(o.po) || throw(ArgumentError("mismatch in number of samples and distributions"))

    return sum([logpdf(msd.sds[program_id], po) for (programid, po) in o])
end

pdf(msd::MultiStudySampleDistribution{T}, o::EvalObservation) where {T <: AbstractProgramState} = exp(logpdf(msd, o)) 
