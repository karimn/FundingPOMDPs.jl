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

function SelectProgramSubsetActionSetFactory(nprograms, nimplement)
    actionlist = map(Combinatorics.combinations(1:nprograms, nimplement)) do programidx
        ImplementOnlyAction(programidx)
    end

    push!(actionlist, ImplementOnlyAction()) # Add a do-nothing action

    return SimpleActionSetFactory(KBanditActionSet(actionlist))
end

function SeparateImplementAndEvalActionSetFactory(nprograms) 
    actionlist = map(Combinatorics.permutations(1:nprograms, 2)) do programidx
        SeparateImplementEvalAction(programidx[1:1], programidx[2:2])
    end

    sameproglist = [SeparateImplementEvalAction([programidx], [programidx]) for programidx in 1:nprograms]

    return SimpleActionSetFactory(KBanditActionSet(vcat(actionlist, sameproglist)))
end

struct ExploreOnlyActionSetFactory{R <: AbstractRewardModel} <: AbstractActionSetFactory{SeparateImplementEvalAction}
    actiondict::Dict{ImplementOnlyAction, KBanditActionSet{SeparateImplementEvalAction}}
    rewardmodel::R
end

function ExploreOnlyActionSetFactory(nprograms, nimplement, neval, rewardmodel)
    actiondict = Dict{ImplementOnlyAction, KBanditActionSet{SeparateImplementEvalAction}}()

    eval_idx_comb = Combinatorics.combinations(1:nprograms, neval)

    for implement_idx in Combinatorics.combinations(1:nprograms, nimplement)        
        key_impl_action = ImplementOnlyAction(implement_idx)

        actiondict[key_impl_action] = KBanditActionSet([SeparateImplementEvalAction(implement_idx, eval_idx) for eval_idx in eval_idx_comb])
    end
   
    no_impl_action = ImplementOnlyAction()
    actiondict[no_impl_action] = KBanditActionSet([SeparateImplementEvalAction(Int64[], eval_idx) for eval_idx in eval_idx_comb])
 
    return ExploreOnlyActionSetFactory(actiondict, rewardmodel)
end

function actions(asf::ExploreOnlyActionSetFactory, s::Union{CausalState, AbstractBelief})
    max_eu = -Inf
    local max_actionset 

    for (impl_action, impl_eval_actionset) in asf.actiondict
        current_eu = expectedutility(asf.rewardmodel, s, impl_action) 

        if current_eu > max_eu
            max_eu = current_eu
            max_actionset = impl_eval_actionset
        end
    end

    return max_actionset
end
