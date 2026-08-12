// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "evo/status.hpp"

namespace evo {

struct SequenceRecord final {
  std::string name;
  std::string bytes;
};

using SequenceRecordCallback =
    std::function<Status(const SequenceRecord &record)>;

// A file beginning with '>' is strict FASTA; every other nonempty file is raw
// bytes. Records are delivered synchronously and released before the next one
// is materialized. The byte limit is enforced while each record is read.
[[nodiscard]] Status
stream_sequence_file(const std::string &path, std::size_t max_record_bytes,
                     const SequenceRecordCallback &callback);

// A file beginning with '>' is strict FASTA; every other nonempty file is raw
// text. Compatibility helper; production inference should use
// stream_sequence_file.
[[nodiscard]] Status read_sequence_file(const std::string &path,
                                        std::vector<SequenceRecord> *records);

} // namespace evo
