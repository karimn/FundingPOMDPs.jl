struct ProgramEvalObservation <: AbstractProgramEvalObservation
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

getprogramid(d::StudySampleDistribution) = getprogramid(d.programstate)
Base.rand(rng::Random.AbstractRNG, d::StudySampleDistribution) = rand(rng, d.programstate, d.samplesize) 
logpdf(d::StudySampleDistribution, ds::StudyDataset) = logpdf(d.programstate, ds)
POMDPs.pdf(d::StudySampleDistribution, ds::StudyDataset) = pdf(d.programstate, ds)
logpdf(d::StudySampleDistribution, o::ProgramEvalObservation) = logpdf(d, o.d)
POMDPs.pdf(d::StudySampleDistribution, o::ProgramEvalObservation) = pdf(d, o.d)

struct MultiStudySampleDistribution{T <: AbstractProgramState} <: AbstractSampleDistribution
    sds::Dict{Int64, StudySampleDistribution{T}}
end

MultiStudySampleDistribution{T}(sds::Vector{StudySampleDistribution{T}}) where{T <: AbstractProgramState} = MultiStudySampleDistribution{T}(Dict(getprogramid(d) => d for d in sds))
Base.rand(rng::Random.AbstractRNG, msd::MultiStudySampleDistribution{T}) where {T <: AbstractProgramState} = EvalObservation{T}([rand(rng, d) for d in msd.sds])

function logpdf(msd::MultiStudySampleDistribution, o::EvalObservation) 
    length(msd.sds) == length(o.po) || throw(ArgumentError("mismatch in number of samples and distributions"))

    return sum([logpdf(msd.sds[program_id], po) for (programid, po) in o])
end

POMDPs.pdf(msd::MultiStudySampleDistribution, o::EvalObservation) = exp(logpdf(msd, o)) 
