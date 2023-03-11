implements(a::AbstractFundingAction, pb::ProgramBelief) = implements(a, pb.pid)
implements(a::AbstractFundingAction, progdgp::ProgramDGP) = implements(a, progdgp.programid)
implements(a::AbstractFundingAction, pcs::ProgramCausalState) = implements(a, pcs.programid)
evaluates(a::AbstractFundingAction, pb::ProgramBelief) = evaluates(a, pb.pid) 

@with_kw_noshow struct ImplementEvalAction <: AbstractFundingAction
    implement_eval_programs::BitSet = BitSet() 
end

ImplementEvalAction(pids::Vector{Int64}) = ImplementEvalAction(BitSet(pids))

implements(a::ImplementEvalAction, i::Int64) = in(i, a.implement_eval_programs)
evaluates(a::ImplementEvalAction, i::Int64) = implements(a, i) 
get_evaluated_program_ids(a::ImplementEvalAction) = a.implement_eval_programs

Base.show(io::IO, a::ImplementEvalAction) = print(io, "ImplementEvalAction([$(a.implement_eval_programs)])")  

@with_kw_noshow struct ImplementOnlyAction <: AbstractFundingAction
    implement_programs::BitSet = BitSet() 
end

ImplementOnlyAction(pids::Vector{Int64}) = ImplementOnlyAction(BitSet(pids))

implements(a::ImplementOnlyAction, i::Int64) = in(i, a.implement_programs)
evaluates(a::ImplementOnlyAction, i::Int64) = false 
get_evaluated_program_ids(::ImplementOnlyAction) = BitSet()

Base.isempty(a::ImplementOnlyAction) = isempty(a.implement_programs) 

Base.show(io::IO, a::ImplementOnlyAction) = print(io, "ImplementOnlyAction([$(a.implement_programs)])")  

Base.length(as::AbstractActionSet) = numactions(as)

@with_kw_noshow struct SeparateImplementEvalAction <: AbstractFundingAction
    implement_programs::BitSet = BitSet() 
    eval_programs::BitSet = BitSet() 
end

SeparateImplementEvalAction(implement_pids::Vector{Int64}, eval_pids::Vector{Int64}) = SeparateImplementEvalAction(BitSet(implement_pids), BitSet(eval_pids))

implements(a::SeparateImplementEvalAction, i::Int64) = in(i, a.implement_programs)
evaluates(a::SeparateImplementEvalAction, i::Int64) = in(i, a.eval_programs) 
get_evaluated_program_ids(a::SeparateImplementEvalAction) = a.eval_programs

Base.show(io::IO, a::SeparateImplementEvalAction) = print(io, "SeparateImplementEvalAction([$(a.implement_programs), $(a.eval_programs)])")  

#=struct ExploreAction <: AbstractFundingAction
    eval_programs::BitSet 
    implement_actionset::AbstractActionSet
end

ExploreAction(pids::Vector{Int64}, impl_as::AbstractActionSet) = ExploreAction(BitSet(pids), impl_as)

evaluates(a::ExploreAction, i::Int64) = in(i, a.eval_programs) 
get_evaluated_program_ids(a::ExploreAction) = a.implement_eval_programs

function expectedutility(r::ExponentialUtilityModel, b::FullBayesianBelief, a::ExploreAction) 
    sum(expectedutility(r, pb, a) for pb in b.progbeliefs)
end

Base.show(io::IO, a::ImplementEvalAction) = print(io, "ImplementEvalAction([$(a.implement_eval_programs)])")  
=#