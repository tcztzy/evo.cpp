# EVO2C model format v1

All integers use little-endian encoding. All offsets are absolute file offsets.
Readers must reject non-canonical offsets, non-zero reserved bytes, arithmetic
overflow, duplicate names, overlapping payloads, and checksum mismatches before
allocating CUDA memory.

## File layout

```text
128-byte header
metadata section
zero padding to 64-byte boundary
N × 256-byte tensor descriptors
zero padding to 64-byte boundary
64-byte-aligned tensor payloads
```

Gaps between tensor payloads are allowed. Every payload offset must be 64-byte
aligned. Header `file_size` must exactly match the mapped regular file.

## Header: 128 bytes

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `u8[8]` | `EVO2C\0\0\0` |
| 8 | `u32` | format version, `1` |
| 12 | `u32` | endian marker, `0x01020304` |
| 16 | `u32` | header size, `128` |
| 20 | `u32` | flags, `0` |
| 24 | `u64` | exact file size |
| 32 | `u64` | metadata offset, `128` |
| 40 | `u64` | metadata size |
| 48 | `u64` | tensor table offset |
| 56 | `u64` | tensor count |
| 64 | `u32` | descriptor size, `256` |
| 68 | `u32` | alignment, `64` |
| 72 | `u64` | data section offset |
| 80 | `u32` | CRC32 of full header with this field zeroed |
| 84 | `u8[44]` | zero |

## Metadata section

Metadata header is 16 bytes: magic `META`, `u16 version=1`, `u16 reserved=0`,
`u32 entry_count`, then `u32 CRC32` over all bytes after the header.

Each entry contains `u16 key_length`, `u8 type`, `u8 reserved=0`,
`u32 value_length`, key bytes, value bytes, then zero padding to an 8-byte
boundary relative to section start. Keys are unique ASCII identifiers using
`[A-Za-z0-9._-]`.

| Type | ID | Encoding |
| --- | ---: | --- |
| string | 1 | raw UTF-8 bytes |
| u64 | 2 | exactly 8 bytes |
| f64 | 3 | IEEE-754, exactly 8 bytes |
| bool | 4 | one byte, `0` or `1` |
| u64 list | 5 | zero or more packed `u64` values |
| bytes | 6 | opaque bytes |

Metadata section ends immediately after the final entry's padding. Maximum
section size is 16 MiB and maximum entry count is 4096.

New multi-size files remain format version 1 and add registry metadata:
`model.id`, pinned architecture/source revisions, source precision, and
`hyena_projection_weight_dtype`. These entries are additive. Legacy 40B v1
files without `model.id` remain loadable when their complete topology,
precision, RoPE, and filter signature unambiguously matches `evo2_40b` or
`evo2_40b_bionemo_bf16`; inconsistent or ambiguous legacy metadata is
rejected with a reconversion diagnostic.

## Tensor descriptor: 256 bytes

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | `char[96]` | NUL-terminated unique tensor name; tail zero |
| 96 | `u8` | dtype |
| 97 | `u8` | rank, 0 through 8 |
| 98 | `u16` | flags, `0` |
| 100 | `u32` | reserved, `0` |
| 104 | `u64[8]` | dimensions; unused entries zero |
| 168 | `u64` | payload offset |
| 176 | `u64` | payload size |
| 184 | `u64` | element count, exact shape product |
| 192 | `u32` | payload CRC32 |
| 196 | `u32` | descriptor CRC32 with this field zeroed |
| 200 | `u8[56]` | zero |

Scalar tensors use rank zero and element count one. Tensor names use the same
identifier alphabet as metadata keys.

| Dtype | ID | Payload size |
| --- | ---: | --- |
| `F32` | 1 | `elements × 4` |
| `BF16` | 2 | `elements × 2` |
| `Q8_0` | 3 | 32-element blocks: BF16 scale + 32 signed bytes (`34` bytes) |
| `E4M3_SW` | 4 | `elements` bytes; scale stored as separate tensor/metadata |

## Validation order

1. Map regular file read-only.
2. Validate header fields, canonical section offsets, padding, and header CRC.
3. Validate metadata CRC, entry types, lengths, keys, and padding.
4. Validate every descriptor CRC, name, dtype, shape product, size, range,
   alignment, uniqueness, and cross-tensor non-overlap.
5. Stream every payload and verify its CRC.
6. Only after success may runtime allocate device memory or transfer weights.
