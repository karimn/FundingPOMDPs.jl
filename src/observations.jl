struct ProgramEvalObservation <: AbstractProgramEvalObservation
    d::StudyDataset
    programid::Int64
end

getprogramid(o::ProgramEvalObservation) = o.programid

getdataset(o::ProgramEvalObservation) = o.d

struct EvalObservation <: AbstractEvalObservation
    programobs::Dict{Int64, AbstractProgramEvalObservation}
end

EvalObservation(pos::Vector{AbstractProgramEvalObservation}) = EvalObservation(Dict(getprogramid(po) => po for po in pos))

getprogramids(o::EvalObservation) = keys(o.programobs)

getdatasets(obs::EvalObservation) = Dict(pid => getdataset(o) for (pid, o) in obs.programobs)

numprograms(obs::EvalObservation) = length(obs.programobs)

struct StudySampleDistribution <: AbstractStudySampleDistribution
    programstate::AbstractProgramState
    samplesize::Int64
end

getprogramid(d::StudySampleDistribution) = getprogramid(d.programstate)

Base.rand(rng::Random.AbstractRNG, d::StudySampleDistribution) = Base.rand(rng, d.programstate, d.samplesize) 
logpdf(d::StudySampleDistribution, ds::StudyDataset) = logpdf(d.programstate, ds)
Distributions.pdf(d::StudySampleDistribution, ds::StudyDataset) = Distributions.pdf(d.programstate, ds)
logpdf(d::StudySampleDistribution, o::ProgramEvalObservation) = logpdf(d, o.d)
Distributions.pdf(d::StudySampleDistribution, o::ProgramEvalObservation) = Distributions.pdf(d, o.d)

struct MultiStudySampleDistribution <: AbstractSampleDistribution 
    sds::Dict{Int64, AbstractStudySampleDistribution}
end

MultiStudySampleDistribution(sds::Vector{AbstractStudySampleDistribution})= MultiStudySampleDistribution(Dict(getprogramid(d) => d for d in sds))
Base.rand(rng::Random.AbstractRNG, msd::MultiStudySampleDistribution) = EvalObservation(Dict(pid => Base.rand(rng, d) for (pid, d) in msd.sds))

function logpdf(msd::MultiStudySampleDistribution, o::EvalObservation) 
    length(msd.sds) == length(o.programobs) || throw(ArgumentError("mismatch in number of samples and distributions"))

    return sum([logpdf(msd.sds[programid], po) for (programid, po) in o.programobs])
end

Distributions.pdf(msd::MultiStudySampleDistribution, o::AbstractEvalObservation) = exp(logpdf(msd, o)) 
