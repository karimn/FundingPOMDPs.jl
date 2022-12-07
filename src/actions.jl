using Parameters

@with_kw struct ImplementEvalAction <: AbstractFundingAction
    implement_eval_programs::BitSet = BitSet() 
end

ImplementEvalAction(pids::Vector{Int64}) = ImplementEvalAction(BitSet(pids))

implements(a::ImplementEvalAction, i::Int64) = in(a.implement_eval_programs, i)
get_evaluated_program_ids(a::ImplementEvalAction) = a.implement_eval_programs
