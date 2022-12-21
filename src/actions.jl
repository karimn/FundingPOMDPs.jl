@with_kw struct ImplementEvalAction <: AbstractFundingAction
    implement_eval_programs::BitSet = BitSet() 
end

ImplementEvalAction(pids::Vector{Int64}) = ImplementEvalAction(BitSet(pids))

implements(a::ImplementEvalAction, i::Int64) = in(i, a.implement_eval_programs)
get_evaluated_program_ids(a::ImplementEvalAction) = a.implement_eval_programs

Base.show(io::IO, a::ImplementEvalAction) = print(io, "ImplementEvalAction([$(a.implement_eval_programs)])")  

@with_kw struct ImplementOnlyAction <: AbstractFundingAction
    implement_programs::BitSet = BitSet() 
end

ImplementOnlyAction(pids::Vector{Int64}) = ImplementOnlyAction(BitSet(pids))

implements(a::ImplementOnlyAction, i::Int64) = in(i, a.implement_programs)
get_evaluated_program_ids(::ImplementOnlyAction) = BitSet() 

Base.show(io::IO, a::ImplementOnlyAction) = print(io, "ImplementOnlyAction([$(a.implement_programs)])")  

Base.length(as::AbstractActionSet) = numactions(as)