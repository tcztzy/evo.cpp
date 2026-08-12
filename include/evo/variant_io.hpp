// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

#include "evo/status.hpp"

namespace evo {

struct VcfRecord final {
  std::string contig;
  std::size_t position_1based{0};
  std::string id;
  std::string reference;
  std::string alternate;
  std::size_t record_index{0};
  std::size_t allele_index{0};
  std::size_t line_number{0};
};

using VcfRecordCallback = std::function<Status(const VcfRecord &record)>;

// Streams one callback per ALT allele. Plain/gzip input and path "-" are
// supported. Symbolic/breakend alleles are outside the DNA scoring contract.
[[nodiscard]] Status stream_vcf_file(const std::string &path,
                                     const VcfRecordCallback &callback);

struct ReferenceSlice final {
  std::string contig;
  std::string sequence;
  std::size_t start{0};
  std::size_t end{0};
  std::size_t contig_length{0};
};

// Scans a plain/gzip FASTA without materializing an entire contig. Two bounded
// passes first determine the exact boundary-aware window and then extract it.
[[nodiscard]] Status fetch_reference_slice(
    const std::string &path, std::string_view contig,
    std::size_t position_1based, std::string_view reference,
    std::string_view alternate, std::size_t maximum_tokens,
    ReferenceSlice *output);

} // namespace evo
