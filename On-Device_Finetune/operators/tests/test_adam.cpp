#include "optimizers/adam.h"
#include "core/tensor.h"
using namespace ops;
int main() {
    std::vector<TensorPtr> params;
    AdamConfig config(0.001f);
    Adam optimizer(params, config);
    return 0;
}
