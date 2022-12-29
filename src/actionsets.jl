struct KBanditActionSet{A} <: AbstractActionSet where A <: AbstractFundingAction
    actions::Vector{A}
end

numactions(as::KBanditActionSet) = length(as.actions)

Base.iterate(as::KBanditActionSet) = iterate(as.actions)
Base.iterate(as::KBanditActionSet, n) = iterate(as.actions, n)
Base.getindex(as::KBanditActionSet, i) = as.actions[i]

function Base.rand(rng::Random.AbstractRNG, as::KBanditActionSet) 
    actid = Base.rand(rng, 1:numactions(as))

    as.actions[actid] 
end

struct SimpleActionSetFactory{A} <: AbstractActionSetFactory{A}
    actionset::KBanditActionSet{A}
end

actions(asf::SimpleActionSetFactory, ::Any = missing) = asf.actionset

function SelectProgramSubsetActionSetFactory(nprograms, nimplement, ::Type{A}) where A <: AbstractFundingAction
    actionlist = map(Combinatorics.combinations(1:nprograms, nimplement)) do programidx
        A(programidx)
    end

    push!(actionlist, A()) # Add a do-nothing action

    return SimpleActionSetFactory(KBanditActionSet(actionlist))
end

function SeparateImplementAndEvalActionSetFactory(nprograms, ::Type{A}) where A <: AbstractFundingAction
    actionlist = map(Combinatorics.permutations(1:nprograms, 2)) do programidx
        A(programidx[1:1], programidx[2:2])
    end

    sameproglist = [A([programidx], [programidx]) for programidx in 1:nprograms]

    return SimpleActionSetFactory(KBanditActionSet(vcat(actionlist, sameproglist)))
end
