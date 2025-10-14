#include "activation_checkpointer.h"

namespace ops {
namespace memory {

ActivationCheckpointer::ActivationCheckpointer(const CheckpointConfig& /*config*/) {
    // Minimal stub to satisfy linker; full implementation is optional for basic checkpoint wrapper
}

ActivationCheckpointer::~ActivationCheckpointer() = default;

} // namespace memory
} // namespace ops


