
abstract type AbstractFundingAction end
abstract type AbstractActionSet{T <: AbstractProgramState} end

struct ImplementEvalAction{T <: AbstractProgramState} <: AbstractFundingAction
    implement_eval_programs::Dict{Int64, T}
end

function ImplementEvalAction{T}() where {T <: AbstractProgramState}
    ImplementEvalAction{T}(Dict{Int64, T}())
end

implements(a::ImplementEvalAction{T}, i::Int64) where {T <: AbstractProgramState} = hashash(a.implement_eval_programs, i)
get_evaluated_programs(a::ImplementEvalAction{T}) where {T <: AbstractProgramState} = a.implement_eval_programs
