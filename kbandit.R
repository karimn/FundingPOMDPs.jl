KBanditActionSet <- R6Class(
  "KBanditActionSet", 
  inherit = ActionSet,

  public = list(
    initialize = function(k, last_action = NULL) {
      map(seq(k), Action$new, action_set = self, last_action = last_action) %>% 
        super$initialize()
    },
    
    get_next_action_set = function(last_action) KBanditActionSet$new(self$k, last_action) 
  ),

  active = list(
    k = function() nrow(private$action_list)
  )
)
