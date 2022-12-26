@with_kw struct ImplementEvalAction <: AbstractFundingAction
    implement_eval_programs::BitSet = BitSet() 
end

ImplementEvalAction(pids::Vector{Int64}) = ImplementEvalAction(BitSet(pids))

implements(a::ImplementEvalAction, i::Int64) = in(i, a.implement_eval_programs)
evaluates(a::ImplementEvalAction, i::Int64) = implements(a, i) 
get_evaluated_program_ids(a::ImplementEvalAction) = a.implement_eval_programs

Base.show(io::IO, a::ImplementEvalAction) = print(io, "ImplementEvalAction([$(a.implement_eval_programs)])")  

@with_kw struct ImplementOnlyAction <: AbstractFundingAction
    implement_programs::BitSet = BitSet() 
end

ImplementOnlyAction(pids::Vector{Int64}) = ImplementOnlyAction(BitSet(pids))

implements(a::ImplementOnlyAction, i::Int64) = in(i, a.implement_programs)
evaluates(a::ImplementOnlyAction, i::Int64) = false 
get_evaluated_program_ids(::ImplementOnlyAction) = BitSet() 

Base.show(io::IO, a::ImplementOnlyAction) = print(io, "ImplementOnlyAction([$(a.implement_programs)])")  

Base.length(as::AbstractActionSet) = numactions(as)

@with_kw struct SeparateImplementEvalAction <: AbstractFundingAction
    implement_programs::BitSet = BitSet() 
    eval_programs::BitSet = BitSet() 
end

SeparateImplementEvalAction(implement_pids::Vector{Int64}, eval_pids::Vector{Int64}) = SeparateImplementEvalAction(BitSet(implement_pids), BitSet(eval_pids))

implements(a::SeparateImplementEvalAction, i::Int64) = in(i, a.implement_programs)
evaluates(a::SeparateImplementEvalAction, i::Int64) = in(i, a.eval_programs) 
get_evaluated_program_ids(a::SeparateImplementEvalAction) = a.eval_programs

Base.show(io::IO, a::SeparateImplementEvalAction) = print(io, "SeparateImplementEvalAction([$(a.implement_programs), $(a.eval_programs)])")  

Base.length(as::SeparateImplementEvalAction) = throw(ErrorException("invalid call for this action type")) 