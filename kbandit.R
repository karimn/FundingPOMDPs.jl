KBanditAction <- R6Class(
  "KBanditAction",
  inherit = Action,

  public = list(
    initialize = function(program_index) {
      private$program_index <- program_index
    },
    
    calculate_expected_simulated_future_value = function(belief, discount, depth) { 
      belief$get_sampled_future_beliefs(self) %>%
        map_dbl(~ .x$expand(discount, depth - 1)) %>%
        mean()
    }, 
    
    calculate_reward = function(belief) {
      belief$calculate_expected_reward(private$program_index)
    }
  ),

  active = list(
    active_program_index = function() private$program_index,
    evaluated_programs = function() private$program_index
  )
)

KBanditActionSet <- R6Class(
  "KBanditActionSet", 
  inherit = ActionSet,

  public = list(
    initialize = function(k) {
      map(seq(k), KBanditAction$new) %>% 
        super$initialize()
    },
    
    get_next_action_set = function(last_action) KBanditActionSet$new(self$k) 
  ),

  active = list(
    k = function() nrow(private$action_list)
  )
)

KBandit <- R6Class(
  "KBandit",
  
  public = list(
    initialize = function(environment) {
      private$env <- environment
    }
  ),
  
  active = list(
    k = function() private$env$num_programs
  ),
  
  private = list(
    env = NULL
  )
)
