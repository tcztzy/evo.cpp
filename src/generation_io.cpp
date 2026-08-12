// SPDX-License-Identifier: Apache-2.0
#include "evo/generation_io.hpp"

#include "evo/variant.hpp"

#include <algorithm>

namespace evo {

Status write_generated_sequence(std::ostream &output,
                                const std::string_view sequence,
                                const GenerationOutputFormat format,
                                const std::string_view name) {
  if (format == GenerationOutputFormat::kRaw) {
    output.write(sequence.data(), static_cast<std::streamsize>(sequence.size()));
  } else {
    if (name.empty() || name.find_first_of("\r\n") != std::string_view::npos) {
      return {ErrorCode::kInvalidArgument,
              "FASTA generation name must be nonempty and single-line"};
    }
    auto status = validate_iupac_dna(sequence, "generated FASTA sequence");
    if (!status.ok())
      return status;
    output << '>' << name << '\n';
    constexpr std::size_t kFastaColumns = 80;
    for (std::size_t offset = 0; offset < sequence.size();
         offset += kFastaColumns) {
      const auto length = std::min(kFastaColumns, sequence.size() - offset);
      output.write(sequence.data() + static_cast<std::ptrdiff_t>(offset),
                   static_cast<std::streamsize>(length));
      output.put('\n');
    }
  }
  output.flush();
  return output ? Status::Ok()
                : Status{ErrorCode::kIo,
                         "failed writing generated sequence to stdout"};
}

} // namespace evo
