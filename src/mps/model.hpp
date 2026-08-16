// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>

#include "evo/cpu/model.hpp"
#include "evo/model_format.hpp"
#include "evo/status.hpp"

namespace evo::detail {
class LinearExecutor;
}

namespace evo::mps {

[[nodiscard]] Status
create_linear_executor(std::shared_ptr<evo::detail::LinearExecutor> *executor);

class ModelLoader final {
public:
  [[nodiscard]] static Status load(const ModelFile &artifact,
                                   bool allow_test_fixture, cpu::Model *model);
};

} // namespace evo::mps
