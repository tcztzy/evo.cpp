// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

#include "evo2c/status.hpp"

namespace evo2c {

struct SequenceRecord final {
  std::string name;
  std::string bytes;
};

// A file beginning with '>' is strict FASTA; every other nonempty file is raw text.
[[nodiscard]] Status read_sequence_file(const std::string& path,
                                        std::vector<SequenceRecord>* records);

}  // namespace evo2c
