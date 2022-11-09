KBanditAction <- R6Class(
  "KBanditAction",
  inherit = Action,

  public = list(
    initialize = function(program_index, last_action) {
      private$program_index <- program_index
      super$initialize(last_action)
    }, 
    
    expand = function(belief, discount, depth) {
      # cat(sprintf("Depth = %d, Action %d\n", depth, private$program_index))
      # cat(sprintf("Expanding (%s)\n", str_c(self$action_trajectory, collapse = ", ")))
      super$expand(belief, discount, depth)
    },
    
    calculate_reward = function(belief) {
      belief$calculate_expected_reward(private$program_index)
    },
    
    calculate_reward_range = function(belief, width = 0.8) {
      belief$calculate_expected_reward_range(private$program_index, width)
    } 
  ),

  active = list(
    active_program_index = function() private$program_index,
    evaluated_programs = function() private$program_index,
    
    action_trajectory = function() { 
      if (is_null(private$last_action)) {
        private$program_index 
      } else {
        c(private$last_action$action_trajectory, private$program_index)
      }
    }
  ),
  
  private = list(
    program_index = NULL
  )
)

KBanditActionSet <- R6Class(
  "KBanditActionSet", 
  inherit = ActionSet,

  public = list(
    initialize = function(k, last_action = NULL) {
      map(seq(k), KBanditAction$new, last_action) %>% 
        super$initialize()
    },
    
    get_next_action_set = function(last_action) KBanditActionSet$new(self$k, last_action) 
  ),

  active = list(
    k = function() nrow(private$action_list)
  )
)
