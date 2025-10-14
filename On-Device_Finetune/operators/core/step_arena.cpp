/**
 * @file step_arena.cpp
 * @brief Step-level arena implementation
 */

#include "step_arena.h"

namespace ops {

StepArena& get_step_arena() {
    static StepArena arena(64);  // 64MB arena
    return arena;
}

} // namespace ops

