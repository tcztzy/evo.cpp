// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "evo/status.hpp"

namespace evo {

enum class SequenceFormat { kRaw, kFasta, kFastq };

struct SequenceRecord final {
  std::string name;
  std::string bytes;
  SequenceFormat format{SequenceFormat::kRaw};
};

using SequenceRecordCallback =
    std::function<Status(const SequenceRecord &record)>;

// Content beginning with '>' is strict FASTA, content beginning with '@' is
// strict four-line FASTQ, and every other nonempty input is raw bytes. Plain
// and gzip files are detected transparently; path "-" reads stdin. Records are
// delivered synchronously and released before the next one is materialized.
// The byte limit is enforced while each record is read.
[[nodiscard]] Status
stream_sequence_file(const std::string &path, std::size_t max_record_bytes,
                     const SequenceRecordCallback &callback);

[[nodiscard]] const char *sequence_format_name(SequenceFormat format) noexcept;

} // namespace evo
