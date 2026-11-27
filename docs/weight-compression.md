# Lossless weight compression

This page measures bit-exact compression of GBCformer's compiled checkpoints, with trained models and synthetic tensors as contrast. Every number was measured on an Apple M1 Max (10 cores, 64 GB) that other jobs shared, so each table gives the load average at the time of the run. All runs use 4 threads unless a row says otherwise. The code was in `tools/compress/`, a Cargo project of its own outside the gbcformer crate.

The tool that produced these numbers, `tools/compress`, has since been removed; the commands under [Reproduce](#reproduce) are kept as the record and no longer run.

## Summary

The compiled checkpoint compresses by four orders of magnitude, and decoding it costs about as much as writing zeros. The 4.18 GB one-model GGUF becomes a 274,046-byte container, 15,238 times smaller, header and metadata included. Rebuilding the GGUF byte for byte takes 0.14 s into memory and 0.35 to 0.49 s into a fresh file before `msync` (measured, 4 threads). Trained weights reach the usual ceiling of about 1.5 times. On Gemma 4 E4B (bf16, 15.0 GB), the same container format gives 1.506 times on the whole file and rebuilds it into memory in 4.8 s (3.1 GB/s on 4 threads). On the GPU, a Metal kernel expands the one-model from an 8.6 MB resident form into its 4.17 GB of tensors in 12.3 ms, 341 GB/s against the chip's 400 GB/s peak.

| method | one-model ratio | cpu-memory ratio | Gemma 3 1B f16 | Gemma 4 E4B bf16 | decode MB/s per core (one-model, Gemma 4) | encode MB/s per core (one-model, Gemma 4) | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gated-per-tensor` (recommended) | 15,824 | 5,053 | 1.49 | 1.50 (full file) | 84,608 / 1,993 | 821 / 639 | 0 |
| `sp-runs-dict+zstd19` | 16,093 | 4,979 | 1.52 | 1.50 (layer 20) | 71,545 / 364 | 1,507 / 2 | 0 |
| `adaptive-per-tensor` | 16,182 | 5,077 | 1.52 | 1.50 (layer 20) | 68,474 / 416 | 156 / 1 | 0 |
| `bytes+zstd19L` | 4,548 | 2,292 | 1.31 | 1.29 (layer 20) | 12,792 / 620 | 71 / 2 | 0 |
| `bytes+zstd3` | 2,816 | 1,559 | 1.30 | 1.28 (full file) | 8,408 / 1,082 | 4,679 / 891 | 0 |
| `zipnn-reimpl` | 23.1 | 22.9 | 1.50 | 1.50 (full file) | 3,063 / 2,037 | 902 / 1,003 | 0 |
| `expmant+huf` | 15.6 | 15.8 | 1.49 | 1.50 (full file) | 1,552 / 1,936 | 675 / 881 | 0 |
| `expwin-bf16` (GPU format) | 1.00 | 1.00 | 1.00 | 1.42 (full file) | 13,345 / 1,253 | 13,521 / 158 | 0 |
| `zfp-reversible(f32)` | 227 | 197 | 0.98 | 0.97 (layer 20) | 859 / 147 | 587 / 156 | 0 |

The ratios count tensor bytes only. The container rows in [Whole-file load time](#whole-file-load-time) count the whole file. The per-core decode rates write into a warm buffer, so read them as relative costs, and read the container numbers for absolute load time.

**Validation.** 36,001 tensor round trips across all sets and methods, each decoded 1 or 3 times and compared byte for byte, and 0 were not bit-exact. Every rebuilt GGUF matched its original with 0 differing bytes. One early version of the dictionary sparse forms did fail, as [Validation](#validation) describes.

**Recommendation.** Use `gated-per-tensor` in a whole-file container. A tensor with under 10% nonzero elements gets the sparse form `sp-runs-dict+zstd19`. Any other tensor gets `expmant+huf`, which is ZipNN's method with a slightly different stream split. The rule costs one pass to count nonzeros, needs no search, and comes within 2.3% of the exhaustive per-tensor search on every set. [Recommendation](#recommendation) gives the reasons.

## The data

| set | source | tensors | tensor bytes | dtypes | nonzero elements | distinct nonzero bit patterns |
|---|---|---:|---:|---|---:|---:|
| one-model | `tools/stock-llama/.build/one-model/model.gguf` | 138 | 4,174,848,000 | 107 f16 matrices, 31 f32 norms | 1,644,425 of 2,087,265,280 (0.079%) | 449 |
| cpu-memory | `.../.build/cpu-memory/model.gguf` | 102 | 350,601,216 | f16 matrices, f32 norms | 301,377 of 175,253,504 (0.17%) | 110 |
| Gemma 3 1B instruct | GGUF `sha256-fd3984f0…` (Google) | 340 | 2,000,040,448 | 183 f16, 157 f32 | 999,885,157 of 999,885,952 | 7,343 |
| Gemma 4 E4B instruct | GGUF `sha256-5de0284c…` (Google) | 720 | 15,037,270,184 | 381 bf16, 339 f32 | 7,518,069,280 of 7,518,069,290 | 11,127 |
| synthetic | generated (`src/synth.rs`) | 15 | 109,445,120 | f16, bf16, f32 | | |

GBCformer's weights are compiled, not trained. Each value is a small integer times a power of two, so almost every element is zero and the nonzeros come from a short list of values. The one-model count includes its 31 f32 norm vectors, which are dense. Its f16 matrices hold 1,485,705 nonzeros. `model.safetensors` beside the cpu-memory GGUF holds the same tensors, so it was not run separately. Gemma 3 and Gemma 4 are trained models from Google, used read-only from a local model store and never copied. Gemma 4 E4B's two embedding tables are 6.98 GB of its 15.0 GB.

The synthetic set is clearly not model weights. It has fixed-seed Gaussian tensors (σ = 0.02, rounded to nearest even) in f16, bf16 and f32, and tensors in which 1 element in 64 is replaced by a special value. It also has 99.9%-zero tensors whose nonzeros are all special values, and every 16-bit pattern once (in order and shuffled, as f16 and as bf16). The last three tensors are 1M random f32 bit patterns, an all-zero tensor and an all-negative-zero tensor. The special values are +0, -0, ±inf, quiet and signalling NaNs with several payloads (for f16 `0x7e00 0x7c01 0x7d55 0xffff 0xfe00 0x7fff`), the smallest and largest subnormals of both signs, the largest finite value and the smallest normal.

## Methods

Each method is a transform to one or more byte streams, then an entropy coder applied to each stream on its own.

**Transforms.**
- `bytes` keeps the stored bytes unchanged.
- `bytegroup` puts byte k of every element in stream k. This is ZipNN's byte grouping.
- `expmant` puts the exponent field in one stream and the sign with the mantissa in byte-wide streams. For f16 that is a 5-bit exponent and 11 bits of sign and mantissa in two streams, for bf16 8 and 8, and for f32 8 and 24.
- `bitplane` writes one stream per bit position, 8 elements per byte.
- `xor-prev`, `xor-row` and `xor-tile` XOR each element with the previous element, with the element one row up, or with the element 32 columns to the left in 32×32 tile order. The result is byte grouped.
- `delta-prev` and `delta-lin` take the residual against the previous element, or against the linear extrapolation `2x[i-1] - x[i-2]`. They work on the order-preserving integer map of the float bits (sign-magnitude to unsigned), then zigzag and byte group the residual.
- `dedup-rows` and `dedup-4k` store identical rows, or identical 4 KiB blocks, once. Blocks are hashed and then compared byte for byte, and each row or block gets a u32 reference.
- The sparse forms treat only the all-zero bit pattern as zero, so -0.0 (`0x8000`) is an entry like any other value. `sp-bitmap` is one bit per element plus the nonzero values. `sp-runs` is a varint zero-run before each nonzero plus the values. `sp-csr` is per-row counts and column gaps plus the values. `sp-runs-dict` is the zero runs plus an index into the tensor's table of distinct nonzero values, ranked by count, at 1, 2, 3 or 4 bytes per index by table size. `sp-rowdedup-csr-dict` stores identical rows once and codes the unique rows as CSR with a value table.
- `expwin-bf16` is the GPU format described in [GPU decode](#gpu-decode). It stores other dtypes raw.

**Entropy coders.**
- `zstd1` through `zstd19` are Zstd at that level, from the `zstd` crate (libzstd 1.5). `zstd19L` adds long-distance matching with a 2^27 window. `bytes+zstd3-1MiB` codes 1 MiB chunks independently, and `+dict` adds a dictionary trained on up to 2,000 64 KiB samples of the same model. The dictionary's bytes count once in the model total.
- `huf` is huff0, the Huffman coder from Yann Collet's [FiniteStateEntropy](https://github.com/Cyan4973/FiniteStateEntropy) (at `9f30e09`), in 128 KiB chunks. ZipNN calls this coder.
- `tans` is FSE, the tabled ANS coder from the same library, in 128 KiB chunks. Its symbols are renamed at build time (`csrc/fse_prefix.h`) so they cannot clash with the copies inside libzstd.
- `rans` is our order-0 static rANS (`src/rans.rs`). It keeps a 32-bit state, renormalises a byte at a time, and uses 15-bit frequencies with one table per 1 MiB.

**Adaptive selection.**
- `adaptive-per-tensor` codes each tensor with every candidate and keeps the smallest. Its metadata is one id byte per tensor.
- `adaptive-per-block` does the same per block of whole rows near 1 MiB. Its metadata is an id byte and a varint length per block, which the "select meta B" column reports.
- The candidates are `bytes+zstd9`, `expmant+huf`, `expmant+zstd3`, `bytegroup+zstd9`, `sp-runs-dict+zstd19`, `sp-rowdedup-csr-dict+zstd19` and `bytes+raw`.
- `gated-per-tensor` picks between `sp-runs-dict+zstd19` and `expmant+huf` by density alone, with the same one-byte metadata.

**Baselines.**
- **ZipNN.** ZipNN has a [paper](https://arxiv.org/abs/2411.05239) and [code](https://github.com/zipnn/zipnn) (read at `6009704`). Its C core includes `Python.h` in every file (`csrc/zipnn_core.c`, `data_manipulation_dtype16.c`), so Rust cannot call it without the Python headers. `zipnn-reimpl` is a reimplementation of that core, read from the C source. It codes 256 KiB chunks and applies the exponent bit reorder for bf16 and f32 (`reorder_float_bits_dtype16` and `_dtype32`). It leaves f16 unreordered, as ZipNN's `zipnn.py` does. Then it byte groups, calls `HUF_compress` per stream, and stores a stream raw when Huffman does not get below 0.95 of its size. It links the same huff0 library. It leaves out ZipNN's speed heuristic that stops trying Huffman on a stream type after 10% of chunks fail, which can only make the reimplementation slower, never larger. Our `bytegroup+huf` and `expmant+huf` are the same idea with different chunking and bit placement.
- **ZFP reversible.** [zfp](https://github.com/LLNL/zfp) (at `2da5c4e`) is built with cmake and called through a C shim (`csrc/zfp_shim.c`). zfp has no 16-bit float type, so f16 and bf16 are widened to f32 bit-exactly and narrowed back the same way after decoding. The widening uses integer operations, never a float cast: it normalises f16 subnormals and shifts NaN payloads up unchanged. Matrices are 2D fields of row length by rows, and vectors are 1D. The widened intermediate is twice the original size, and reversible zfp does not earn it back on trained weights (0.97 to 0.98 times).
- **NWC.** NWC is Neural Weight Compression ([arXiv 2510.11234](https://arxiv.org/abs/2510.11234), Ryu et al., ICML 2026). It is a learned autoencoder codec with entropy-constrained quantization, trained on pretrained weights and judged by downstream accuracy at 4 to 6 bits per weight. It has no lossless mode, so it cannot meet a bit-exact requirement. Reimplementing it would mean training a neural codec in PyTorch, which this project does not allow. It is not run here.

## Validation

The harness round-trips every tensor of every set through every method and compares raw bytes (`out == original`, a memcmp), never float values. Before each decode it fills the target with `0xA5`, so a decoder that skips a byte fails. A sparse decoder that assumes zeroed memory is the case this catches. The container check rebuilds each GGUF and compares the whole file byte for byte.

The first full synthetic run (`synth-v1-devrun.tsv`, kept with the results) failed 21 round trips. All seven dictionary methods failed on the same three f32 tensors (`gauss-f32`, `special-f32` and `random-bits-f32`). The index width was 2 bytes for any table above 256 values, and those tensors have more than 65,536 distinct values. The fix chooses 1, 2, 3 or 4 bytes from the table size (`index_width` in `src/sparse.rs`). Every run since then has 0 failures. A second bug did not corrupt data but did invalidate a row: on Gemma 3 and Gemma 4 layer 20, the `--skip` filter switched off dictionary training, so `bytes+zstd3-1MiB+dict` first ran with an empty dictionary. Those two rows were rerun after the fix (`want_dict` in `src/main.rs`).

| set | method rows | tensor round trips | not bit-exact |
|---|---:|---:|---:|
| synth | 46 | 690 | 0 |
| cpu-memory | 46 | 4692 | 0 |
| one-model | 46 | 6350 | 0 |
| gemma3-1b-f16 | 47 | 16010 | 0 |
| gemma4-e4b-bf16-blk20 | 47 | 799 | 0 |
| gemma4-e4b-bf16-full | 10 | 7460 | 0 |
| synth, first run (superseded) | 44 | 660 | 21 |

The container check rebuilt each of the four GGUFs into a file 3 times and byte-compared the last rebuild with the original, with 0 differing bytes in 21,586,275,104. The rebuilds into memory were timed but not compared. They run the same decoders. The Metal runs byte-compared every tensor after every run: 0 mismatches.

The special-value tensors pass on every method, ZFP's f32 path included. That covers every f16 and bf16 bit pattern, NaN payloads, signalling NaNs, ±0, ±inf and subnormals.

## Results

Ratio is original tensor bytes over compressed bytes. The compressed bytes include every stream header, the selection metadata and, for `+dict`, the dictionary. Throughput is per core: tensor bytes divided by the summed per-tensor encode or decode time, the median of the reps, with 4 tensors in flight at once. "dec heap peak" is the largest extra Rust heap that any one tensor's decode allocated beyond its output buffer. zstd, huff0 and zfp allocate in C, and those allocations are not counted.

Three effects matter for reading the tables.
- Each decode writes into a buffer the harness just filled with `0xA5`, so tensors that fit in the 48 MB system-level cache decode at cache speed. That is why sparse decodes show 60 to 85 GB/s per core on the GBC models. Treat them as relative costs. The container section measures the cold path.
- The heap peaks of 2 to 9 times a tensor's size on the dense models come from this implementation, which widens to u32 and copies whole streams. The formats do not need it: `zipnn-reimpl` decodes in 256 KiB chunks with a 0.8 MB peak. Peak RSS of the container rebuild is what a loader pays.
- cpu-memory, one-model and the fast pass on Gemma 3 coded each tensor whole. Later runs code tensors above 256 MiB in 256 MiB pieces of whole rows (`gguf::pieces`), which bounds working memory. Those later runs are the slow pass on Gemma 3, everything on Gemma 4, `gated-per-tensor` everywhere, and the container. Splitting changes one-model's `token_embd` and Gemma 3's embedding, and it is why the tensor counts in the detail tables differ by one or two.

On the dense models the slow encoders (zstd -19, the adaptive searches and the dictionary sparse forms) ran once, and everything else ran 3 times. Gemma 4 ran every method on layer 20 (17 tensors, 186 MB). It also ran the fast methods on the whole 15.0 GB file. `expwin-bf16` and `gated-per-tensor` were added after the first sets ran and have their own rows.

Compression ratio (FAIL n marks a method that failed n tensors).

| method | synth | cpu-memory | one-model | gemma3-1b-f16 | gemma4-e4b-bf16-blk20 | gemma4-e4b-bf16-full |
|---|---:|---:|---:|---:|---:|---:|
| bytes+raw | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |
| bytes+zstd1 | 1.29 | 1490 | 2619 | 1.30 | 1.28 | 1.28 |
| bytes+zstd3 | 1.29 | 1559 | 2816 | 1.30 | 1.28 | 1.28 |
| bytes+zstd9 | 1.30 | 2181 | 4056 | 1.31 | 1.29 |  |
| bytes+zstd19 | 1.30 | 2165 | 4381 | 1.31 | 1.29 |  |
| bytes+zstd19L | 1.30 | 2292 | 4548 | 1.31 | 1.29 |  |
| bytes+huf | 1.27 | 13.74 | 12.92 | 1.30 | 1.28 | 1.28 |
| bytes+tans | 1.29 | 768 | 1303 | 1.31 | 1.28 |  |
| bytes+rans | 1.29 | 683 | 1124 | 1.31 | 1.28 |  |
| bytes+zstd3-1MiB | 1.29 | 1505 | 2631 | 1.30 | 1.28 |  |
| bytegroup+zstd3 | 1.41 | 1771 | 3547 | 1.50 | 1.40 | 1.40 |
| bytes+zstd3-1MiB+dict | 1.29 | 1154 | 2385 | 1.30 | 1.28 |  |
| bytegroup+zstd19 | 1.44 | 2345 | 5009 | 1.50 | 1.48 |  |
| bytegroup+huf | 1.42 | 22.94 | 23.12 | 1.50 | 1.48 | 1.49 |
| bytegroup+rans | 1.44 | 727 | 1216 | 1.52 | 1.49 |  |
| expmant+zstd3 | 1.36 | 1379 | 2763 | 1.34 | 1.41 |  |
| expmant+zstd19 | 1.44 | 1840 | 3816 | 1.49 | 1.49 |  |
| expmant+huf | 1.42 | 15.80 | 15.64 | 1.49 | 1.50 | 1.50 |
| expmant+tans | 1.45 | 794 | 1324 | 1.51 | 1.50 | 1.51 |
| expmant+rans | 1.45 | 570 | 920 | 1.51 | 1.50 |  |
| bitplane+zstd3 | 1.34 | 1128 | 2829 | 1.37 | 1.35 |  |
| bitplane+rans | 1.34 | 312 | 838 | 1.38 | 1.36 |  |
| xor-prev+zstd3 | 1.37 | 1513 | 3284 | 1.41 | 1.38 |  |
| xor-row+zstd3 | 1.36 | 2049 | 4320 | 1.41 | 1.36 |  |
| xor-tile+zstd3 | 1.36 | 1624 | 3889 | 1.40 | 1.35 |  |
| delta-prev+zstd3 | 1.33 | 871 | 1833 | 1.24 | 1.31 |  |
| delta-lin+zstd3 | 1.24 | 1026 | 2194 | 1.09 | 1.20 |  |
| dedup-rows+zstd3 | 1.41 | 1527 | 3358 | 1.50 | 1.40 |  |
| dedup-4k+zstd3 | 1.29 | 1385 | 3332 | 1.30 | 1.28 |  |
| sp-bitmap+zstd3 | 1.41 | 3139 | 8713 | 1.50 | 1.40 |  |
| sp-bitmap+rans | 1.44 | 1089 | 2419 | 1.52 | 1.49 |  |
| sp-runs+zstd3 | 1.41 | 3933 | 11785 | 1.50 | 1.40 |  |
| sp-csr+zstd3 | 1.41 | 3818 | 11173 | 1.50 | 1.40 |  |
| sp-runs-dict+zstd3 | 1.21 | 4003 | 12364 | 1.43 | 1.41 |  |
| sp-runs-dict+zstd19 | 1.33 | 4979 | 16093 | 1.52 | 1.50 |  |
| sp-runs-dict+huf | 1.19 | 1612 | 3092 | 1.51 | 1.50 |  |
| sp-runs-dict+rans | 1.18 | 1168 | 2922 | 1.53 | 1.51 |  |
| sp-rowdedup-csr-dict+zstd3 | 1.21 | 2554 | 7056 | 1.43 | 1.41 |  |
| sp-rowdedup-csr-dict+zstd19 | 1.33 | 2973 | 8186 | 1.52 | 1.50 |  |
| sp-rowdedup-csr-dict+rans | 1.18 | 972 | 2728 | 1.53 | 1.51 |  |
| zipnn-reimpl | 1.43 | 22.94 | 23.12 | 1.50 | 1.50 | 1.50 |
| zfp-reversible(f32) | 1.04 | 197 | 227 | 0.98 | 0.97 |  |
| adaptive-per-tensor | 1.45 | 5077 | 16182 | 1.52 | 1.50 |  |
| adaptive-per-block | 1.45 | 3106 | 5867 | 1.51 | 1.50 |  |
| expwin-bf16 | 1.12 | 1.00 | 1.00 | 1.00 | 1.42 | 1.42 |
| gated-per-tensor | 1.44 | 5053 | 15824 | 1.49 | 1.50 | 1.50 |

Decode MB/s per core (warm destination buffer).

| method | synth | cpu-memory | one-model | gemma3-1b-f16 | gemma4-e4b-bf16-blk20 | gemma4-e4b-bf16-full |
|---|---:|---:|---:|---:|---:|---:|
| bytes+raw | 10202 | 10046 | 12560 | 12208 | 15640 |  |
| bytes+zstd1 | 1281 | 8778 | 7853 | 1174 | 1107 | 1083 |
| bytes+zstd3 | 1287 | 8356 | 8408 | 1170 | 1087 | 1082 |
| bytes+zstd9 | 1212 | 12548 | 10930 | 963 | 885 |  |
| bytes+zstd19 | 968 | 14735 | 13378 | 916 | 615 |  |
| bytes+zstd19L | 990 | 14297 | 12792 | 949 | 620 |  |
| bytes+huf | 1221 | 3009 | 2806 | 1415 | 1332 | 1326 |
| bytes+tans | 562 | 751 | 717 | 545 | 550 |  |
| bytes+rans | 151 | 189 | 191 | 144 | 150 |  |
| bytes+zstd3-1MiB | 1359 | 19442 | 17088 | 1213 | 1125 |  |
| bytegroup+zstd3 | 1679 | 2111 | 2908 | 1260 | 1312 | 1344 |
| bytes+zstd3-1MiB+dict | 1349 | 20550 | 16804 | 1194 | 1098 |  |
| bytegroup+zstd19 | 2628 | 3642 | 3845 | 1181 | 1942 |  |
| bytegroup+huf | 2188 | 2257 | 2207 | 1274 | 2341 | 2051 |
| bytegroup+rans | 155 | 183 | 184 | 150 | 155 |  |
| expmant+zstd3 | 1120 | 1616 | 1934 | 603 | 1359 |  |
| expmant+zstd19 | 2062 | 2596 | 2874 | 902 | 1832 |  |
| expmant+huf | 1739 | 1549 | 1552 | 925 | 2178 | 1936 |
| expmant+tans | 719 | 683 | 688 | 336 | 908 | 874 |
| expmant+rans | 135 | 123 | 123 | 104 | 155 |  |
| bitplane+zstd3 | 169 | 173 | 178 | 173 | 175 |  |
| bitplane+rans | 88 | 92 | 92 | 87 | 89 |  |
| xor-prev+zstd3 | 670 | 713 | 706 | 512 | 557 |  |
| xor-row+zstd3 | 1248 | 1495 | 1511 | 861 | 955 |  |
| xor-tile+zstd3 | 594 | 561 | 519 | 425 | 480 |  |
| delta-prev+zstd3 | 800 | 870 | 844 | 511 | 674 |  |
| delta-lin+zstd3 | 483 | 465 | 458 | 347 | 416 |  |
| dedup-rows+zstd3 | 1933 | 8701 | 8214 | 1198 | 1231 |  |
| dedup-4k+zstd3 | 1317 | 20856 | 32086 | 1168 | 1096 |  |
| sp-bitmap+zstd3 | 650 | 20208 | 22964 | 523 | 545 |  |
| sp-bitmap+rans | 142 | 2607 | 2662 | 123 | 126 |  |
| sp-runs+zstd3 | 441 | 59232 | 71226 | 350 | 349 |  |
| sp-csr+zstd3 | 417 | 59554 | 57782 | 332 | 330 |  |
| sp-runs-dict+zstd3 | 372 | 57742 | 67868 | 319 | 341 |  |
| sp-runs-dict+zstd19 | 374 | 67413 | 71545 | 360 | 364 |  |
| sp-runs-dict+huf | 390 | 68506 | 67824 | 356 | 363 |  |
| sp-runs-dict+rans | 91 | 34312 | 53873 | 87 | 89 |  |
| sp-rowdedup-csr-dict+zstd3 | 339 | 30118 | 31949 | 305 | 310 |  |
| sp-rowdedup-csr-dict+zstd19 | 336 | 28885 | 31300 | 340 | 332 |  |
| sp-rowdedup-csr-dict+rans | 89 | 20678 | 28124 | 86 | 86 |  |
| zipnn-reimpl | 2293 | 3035 | 3063 | 1427 | 2082 | 2037 |
| zfp-reversible(f32) | 151 | 836 | 859 | 133 | 147 |  |
| adaptive-per-tensor | 535 | 60078 | 68474 | 379 | 416 |  |
| adaptive-per-block | 2342 | 9965 | 10458 | 443 | 2206 |  |
| expwin-bf16 | 3575 | 15771 | 13345 | 22587 | 1217 | 1253 |
| gated-per-tensor | 1889 | 73812 | 84608 | 958 | 2071 | 1993 |

Encode MB/s per core.

| method | synth | cpu-memory | one-model | gemma3-1b-f16 | gemma4-e4b-bf16-blk20 | gemma4-e4b-bf16-full |
|---|---:|---:|---:|---:|---:|---:|
| bytes+raw | 5264 | 6127 | 6355 | 7788 | 7764 |  |
| bytes+zstd1 | 1228 | 6566 | 6176 | 1148 | 1064 | 1005 |
| bytes+zstd3 | 886 | 4511 | 4679 | 1022 | 920 | 891 |
| bytes+zstd9 | 203 | 1179 | 1223 | 75 | 80 |  |
| bytes+zstd19 | 5 | 266 | 346 | 3 | 3 |  |
| bytes+zstd19L | 4 | 97 | 71 | 2 | 2 |  |
| bytes+huf | 918 | 1110 | 1058 | 936 | 918 | 879 |
| bytes+tans | 491 | 657 | 628 | 483 | 473 |  |
| bytes+rans | 158 | 149 | 149 | 156 | 160 |  |
| bytes+zstd3-1MiB | 1009 | 6772 | 7166 | 1071 | 966 |  |
| bytegroup+zstd3 | 543 | 1711 | 1803 | 691 | 325 | 324 |
| bytes+zstd3-1MiB+dict | 830 | 6839 | 7091 | 907 | 801 |  |
| bytegroup+zstd19 | 4 | 249 | 322 | 3 | 2 |  |
| bytegroup+huf | 986 | 991 | 992 | 736 | 941 | 917 |
| bytegroup+rans | 155 | 144 | 145 | 158 | 155 |  |
| expmant+zstd3 | 270 | 1200 | 1222 | 157 | 338 |  |
| expmant+zstd19 | 2 | 182 | 225 | 1 | 2 |  |
| expmant+huf | 797 | 669 | 675 | 492 | 887 | 881 |
| expmant+tans | 517 | 501 | 505 | 281 | 628 | 596 |
| expmant+rans | 130 | 97 | 97 | 105 | 154 |  |
| bitplane+zstd3 | 360 | 399 | 404 | 334 | 338 |  |
| bitplane+rans | 123 | 112 | 112 | 121 | 121 |  |
| xor-prev+zstd3 | 433 | 973 | 967 | 537 | 264 |  |
| xor-row+zstd3 | 434 | 977 | 968 | 523 | 252 |  |
| xor-tile+zstd3 | 353 | 603 | 571 | 404 | 219 |  |
| delta-prev+zstd3 | 394 | 683 | 664 | 436 | 251 |  |
| delta-lin+zstd3 | 399 | 598 | 589 | 350 | 277 |  |
| dedup-rows+zstd3 | 393 | 1417 | 1446 | 413 | 242 |  |
| dedup-4k+zstd3 | 502 | 1677 | 2183 | 498 | 529 |  |
| sp-bitmap+zstd3 | 371 | 2337 | 2333 | 369 | 235 |  |
| sp-bitmap+rans | 142 | 1174 | 1181 | 124 | 124 |  |
| sp-runs+zstd3 | 289 | 1783 | 1809 | 271 | 190 |  |
| sp-csr+zstd3 | 284 | 2460 | 2478 | 266 | 183 |  |
| sp-runs-dict+zstd3 | 15 | 1747 | 1765 | 75 | 74 |  |
| sp-runs-dict+zstd19 | 3 | 1231 | 1507 | 2 | 2 |  |
| sp-runs-dict+huf | 16 | 1734 | 1767 | 85 | 90 |  |
| sp-runs-dict+rans | 12 | 1708 | 1761 | 50 | 50 |  |
| sp-rowdedup-csr-dict+zstd3 | 12 | 721 | 747 | 67 | 69 |  |
| sp-rowdedup-csr-dict+zstd19 | 2 | 625 | 695 | 2 | 2 |  |
| sp-rowdedup-csr-dict+rans | 12 | 711 | 741 | 46 | 47 |  |
| zipnn-reimpl | 1069 | 895 | 902 | 808 | 1025 | 1003 |
| zfp-reversible(f32) | 169 | 594 | 587 | 134 | 156 |  |
| adaptive-per-tensor | 1 | 148 | 156 | 1 | 1 |  |
| adaptive-per-block | 3 | 159 | 172 | 3 | 3 |  |
| expwin-bf16 | 441 | 14574 | 13521 | 17725 | 157 | 158 |
| gated-per-tensor | 672 | 721 | 821 | 396 | 661 | 639 |

<details><summary>synth: 15 tensors, 109.4 MB, sorted by size</summary>

| method | ratio | out MB | select meta B | enc MB/s/core | dec MB/s/core | dec heap peak MB | reps | load | failures |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| adaptive-per-tensor | 1.45 | 75.4 | 15 | 1 | 535 | 251.8 | 3 | 4.09 5.40 5.86 | 0 |
| adaptive-per-block | 1.45 | 75.5 | 410 | 3 | 2342 | 7.9 | 3 | 4.85 5.43 5.82 | 0 |
| expmant+rans | 1.45 | 75.6 | 0 | 130 | 135 | 184.5 | 3 | 4.74 6.08 6.15 | 0 |
| expmant+tans | 1.45 | 75.7 | 0 | 517 | 719 | 184.5 | 3 | 4.80 6.12 6.16 | 0 |
| expmant+zstd19 | 1.44 | 75.7 | 0 | 2 | 2062 | 184.5 | 3 | 7.68 6.74 6.36 | 0 |
| bytegroup+zstd19 | 1.44 | 75.9 | 0 | 4 | 2628 | 100.7 | 3 | 4.48 6.18 6.16 | 0 |
| sp-bitmap+rans | 1.44 | 75.9 | 0 | 142 | 142 | 102.8 | 3 | 4.63 6.01 6.12 | 0 |
| bytegroup+rans | 1.44 | 75.9 | 0 | 155 | 155 | 100.7 | 3 | 7.83 6.76 6.37 | 0 |
| gated-per-tensor | 1.44 | 75.9 | 15 | 672 | 1889 | 184.5 | 3 | 4.43 4.42 4.50 | 0 |
| zipnn-reimpl | 1.43 | 76.8 | 0 | 1069 | 2293 | 0.8 | 3 | 4.09 5.40 5.86 | 0 |
| bytegroup+huf | 1.42 | 77.1 | 0 | 986 | 2188 | 100.7 | 3 | 7.83 6.76 6.37 | 0 |
| expmant+huf | 1.42 | 77.1 | 0 | 797 | 1739 | 184.5 | 3 | 4.80 6.12 6.16 | 0 |
| sp-bitmap+zstd3 | 1.41 | 77.8 | 0 | 371 | 650 | 102.8 | 3 | 4.60 6.03 6.13 | 0 |
| sp-runs+zstd3 | 1.41 | 77.8 | 0 | 289 | 441 | 251.7 | 3 | 4.63 6.01 6.12 | 0 |
| sp-csr+zstd3 | 1.41 | 77.8 | 0 | 284 | 417 | 251.7 | 3 | 4.63 6.01 6.12 | 0 |
| bytegroup+zstd3 | 1.41 | 77.8 | 0 | 543 | 1679 | 100.7 | 3 | 4.48 6.18 6.16 | 0 |
| dedup-rows+zstd3 | 1.41 | 77.8 | 0 | 393 | 1933 | 134.3 | 3 | 4.60 6.03 6.13 | 0 |
| xor-prev+zstd3 | 1.37 | 79.8 | 0 | 433 | 670 | 100.7 | 3 | 4.60 6.03 6.13 | 0 |
| expmant+zstd3 | 1.36 | 80.3 | 0 | 270 | 1120 | 184.5 | 3 | 7.83 6.76 6.37 | 0 |
| xor-row+zstd3 | 1.36 | 80.3 | 0 | 434 | 1248 | 100.7 | 3 | 4.60 6.03 6.13 | 0 |
| xor-tile+zstd3 | 1.36 | 80.4 | 0 | 353 | 594 | 302.0 | 3 | 4.60 6.03 6.13 | 0 |
| bitplane+rans | 1.34 | 81.4 | 0 | 123 | 88 | 100.7 | 3 | 4.74 6.08 6.15 | 0 |
| bitplane+zstd3 | 1.34 | 81.6 | 0 | 360 | 169 | 100.7 | 3 | 4.74 6.08 6.15 | 0 |
| sp-runs-dict+zstd19 | 1.33 | 82.3 | 0 | 3 | 374 | 251.8 | 3 | 4.84 6.00 6.11 | 0 |
| sp-rowdedup-csr-dict+zstd19 | 1.33 | 82.3 | 0 | 2 | 336 | 285.3 | 3 | 4.65 5.78 6.03 | 0 |
| delta-prev+zstd3 | 1.33 | 82.5 | 0 | 394 | 800 | 167.8 | 3 | 4.60 6.03 6.13 | 0 |
| bytes+zstd19L | 1.30 | 84.2 | 0 | 4 | 990 | 67.1 | 3 | 3.60 6.34 6.21 | 0 |
| bytes+zstd19 | 1.30 | 84.2 | 0 | 5 | 968 | 67.1 | 3 | 2.95 6.57 6.28 | 0 |
| bytes+zstd9 | 1.30 | 84.3 | 0 | 203 | 1212 | 67.1 | 3 | 2.95 6.63 6.30 | 0 |
| bytes+rans | 1.29 | 84.5 | 0 | 158 | 151 | 33.6 | 3 | 4.61 6.23 6.18 | 0 |
| bytes+zstd1 | 1.29 | 84.6 | 0 | 1228 | 1281 | 67.1 | 3 | 2.95 6.63 6.30 | 0 |
| bytes+zstd3-1MiB | 1.29 | 84.6 | 0 | 1009 | 1359 | 0.0 | 3 | 4.48 6.18 6.16 | 0 |
| bytes+zstd3 | 1.29 | 84.6 | 0 | 886 | 1287 | 67.1 | 3 | 2.95 6.63 6.30 | 0 |
| bytes+zstd3-1MiB+dict | 1.29 | 84.6 | 0 | 830 | 1349 | 0.0 | 3 | 4.48 6.18 6.16 | 0 |
| dedup-4k+zstd3 | 1.29 | 84.7 | 0 | 502 | 1317 | 67.1 | 3 | 4.60 6.03 6.13 | 0 |
| bytes+tans | 1.29 | 84.7 | 0 | 491 | 562 | 33.6 | 3 | 4.61 6.23 6.18 | 0 |
| bytes+huf | 1.27 | 85.9 | 0 | 918 | 1221 | 33.6 | 3 | 4.61 6.23 6.18 | 0 |
| delta-lin+zstd3 | 1.24 | 88.4 | 0 | 399 | 483 | 167.8 | 3 | 4.60 6.03 6.13 | 0 |
| sp-runs-dict+zstd3 | 1.21 | 90.8 | 0 | 15 | 372 | 251.8 | 3 | 4.63 6.01 6.12 | 0 |
| sp-rowdedup-csr-dict+zstd3 | 1.21 | 90.8 | 0 | 12 | 339 | 285.3 | 3 | 5.03 5.92 6.08 | 0 |
| sp-runs-dict+huf | 1.19 | 92.0 | 0 | 16 | 390 | 251.8 | 3 | 5.90 6.15 6.17 | 0 |
| sp-runs-dict+rans | 1.18 | 92.6 | 0 | 12 | 91 | 251.8 | 3 | 5.61 6.08 6.14 | 0 |
| sp-rowdedup-csr-dict+rans | 1.18 | 92.6 | 0 | 12 | 89 | 285.3 | 3 | 4.15 5.50 5.91 | 0 |
| expwin-bf16 | 1.12 | 98.0 | 0 | 441 | 3575 | 57.1 | 3 | 4.38 5.29 5.75 | 0 |
| zfp-reversible(f32) | 1.04 | 105.7 | 0 | 169 | 151 | 67.1 | 3 | 4.09 5.40 5.86 | 0 |
| bytes+raw | 1.00 | 109.4 | 0 | 5264 | 10202 | 33.6 | 3 | 2.95 6.63 6.30 | 0 |

</details>

<details><summary>cpu-memory: 102 tensors, 350.6 MB, sorted by size</summary>

| method | ratio | out MB | select meta B | enc MB/s/core | dec MB/s/core | dec heap peak MB | reps | load | failures |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| adaptive-per-tensor | 5077 | 0.07 | 102 | 148 | 60078 | 1.6 | 3 | 4.59 5.18 5.68 | 0 |
| gated-per-tensor | 5053 | 0.07 | 102 | 721 | 73812 | 1.6 | 3 | 4.43 4.42 4.50 | 0 |
| sp-runs-dict+zstd19 | 4979 | 0.07 | 0 | 1231 | 67413 | 1.6 | 3 | 4.59 5.18 5.68 | 0 |
| sp-runs-dict+zstd3 | 4003 | 0.09 | 0 | 1747 | 57742 | 1.6 | 3 | 4.59 5.18 5.68 | 0 |
| sp-runs+zstd3 | 3933 | 0.09 | 0 | 1783 | 59232 | 1.7 | 3 | 4.59 5.18 5.68 | 0 |
| sp-csr+zstd3 | 3818 | 0.09 | 0 | 2460 | 59554 | 1.8 | 3 | 4.59 5.18 5.68 | 0 |
| sp-bitmap+zstd3 | 3139 | 0.11 | 0 | 2337 | 20208 | 4.2 | 3 | 4.55 5.19 5.69 | 0 |
| adaptive-per-block | 3106 | 0.11 | 932 | 159 | 9965 | 5.8 | 3 | 4.62 5.18 5.68 | 0 |
| sp-rowdedup-csr-dict+zstd19 | 2973 | 0.12 | 0 | 625 | 28885 | 35.7 | 3 | 4.59 5.18 5.68 | 0 |
| sp-rowdedup-csr-dict+zstd3 | 2554 | 0.14 | 0 | 721 | 30118 | 35.7 | 3 | 4.59 5.18 5.68 | 0 |
| bytegroup+zstd19 | 2345 | 0.15 | 0 | 249 | 3642 | 101.7 | 3 | 4.39 5.24 5.73 | 0 |
| bytes+zstd19L | 2292 | 0.15 | 0 | 97 | 14297 | 67.8 | 3 | 4.19 5.23 5.73 | 0 |
| bytes+zstd9 | 2181 | 0.16 | 0 | 1179 | 12548 | 67.8 | 3 | 4.19 5.23 5.73 | 0 |
| bytes+zstd19 | 2165 | 0.16 | 0 | 266 | 14735 | 67.8 | 3 | 4.19 5.23 5.73 | 0 |
| xor-row+zstd3 | 2049 | 0.17 | 0 | 977 | 1495 | 101.7 | 3 | 4.55 5.19 5.69 | 0 |
| expmant+zstd19 | 1840 | 0.19 | 0 | 182 | 2596 | 186.5 | 3 | 4.27 5.18 5.70 | 0 |
| bytegroup+zstd3 | 1771 | 0.20 | 0 | 1711 | 2111 | 101.7 | 3 | 4.39 5.24 5.73 | 0 |
| xor-tile+zstd3 | 1624 | 0.22 | 0 | 603 | 561 | 305.2 | 3 | 4.55 5.19 5.69 | 0 |
| sp-runs-dict+huf | 1612 | 0.22 | 0 | 1734 | 68506 | 1.6 | 3 | 4.59 5.18 5.68 | 0 |
| bytes+zstd3 | 1559 | 0.22 | 0 | 4511 | 8356 | 67.8 | 3 | 4.19 5.23 5.73 | 0 |
| dedup-rows+zstd3 | 1527 | 0.23 | 0 | 1417 | 8701 | 135.2 | 3 | 4.55 5.19 5.69 | 0 |
| xor-prev+zstd3 | 1513 | 0.23 | 0 | 973 | 713 | 101.7 | 3 | 4.51 5.19 5.69 | 0 |
| bytes+zstd3-1MiB | 1505 | 0.23 | 0 | 6772 | 19442 | 0.0 | 3 | 4.39 5.24 5.73 | 0 |
| bytes+zstd1 | 1490 | 0.24 | 0 | 6566 | 8778 | 67.8 | 3 | 4.19 5.23 5.73 | 0 |
| dedup-4k+zstd3 | 1385 | 0.25 | 0 | 1677 | 20856 | 67.6 | 3 | 4.55 5.19 5.69 | 0 |
| expmant+zstd3 | 1379 | 0.25 | 0 | 1200 | 1616 | 186.5 | 3 | 4.20 5.19 5.70 | 0 |
| sp-runs-dict+rans | 1168 | 0.30 | 0 | 1708 | 34312 | 1.6 | 3 | 4.59 5.18 5.68 | 0 |
| bytes+zstd3-1MiB+dict | 1154 | 0.30 | 0 | 6839 | 20550 | 0.0 | 3 | 4.39 5.24 5.73 | 0 |
| bitplane+zstd3 | 1128 | 0.31 | 0 | 399 | 173 | 101.7 | 3 | 4.30 5.16 5.68 | 0 |
| sp-bitmap+rans | 1089 | 0.32 | 0 | 1174 | 2607 | 2.7 | 3 | 4.55 5.19 5.69 | 0 |
| delta-lin+zstd3 | 1026 | 0.34 | 0 | 598 | 465 | 169.6 | 3 | 4.55 5.19 5.69 | 0 |
| sp-rowdedup-csr-dict+rans | 972 | 0.36 | 0 | 711 | 20678 | 35.7 | 3 | 4.59 5.18 5.68 | 0 |
| delta-prev+zstd3 | 871 | 0.40 | 0 | 683 | 870 | 169.6 | 3 | 4.55 5.19 5.69 | 0 |
| expmant+tans | 794 | 0.44 | 0 | 501 | 683 | 186.5 | 3 | 4.27 5.18 5.70 | 0 |
| bytes+tans | 768 | 0.46 | 0 | 657 | 751 | 33.9 | 3 | 4.25 5.23 5.72 | 0 |
| bytegroup+rans | 727 | 0.48 | 0 | 144 | 183 | 101.7 | 3 | 4.20 5.19 5.70 | 0 |
| bytes+rans | 683 | 0.51 | 0 | 149 | 189 | 33.9 | 3 | 4.25 5.23 5.72 | 0 |
| expmant+rans | 570 | 0.62 | 0 | 97 | 123 | 186.5 | 3 | 4.27 5.18 5.70 | 0 |
| bitplane+rans | 312 | 1.12 | 0 | 112 | 92 | 101.7 | 3 | 4.30 5.16 5.68 | 0 |
| zfp-reversible(f32) | 197 | 1.78 | 0 | 594 | 836 | 67.8 | 3 | 4.59 5.18 5.68 | 0 |
| bytegroup+huf | 22.94 | 15.3 | 0 | 991 | 2257 | 101.7 | 3 | 4.20 5.19 5.70 | 0 |
| zipnn-reimpl | 22.94 | 15.3 | 0 | 895 | 3035 | 0.8 | 3 | 4.59 5.18 5.68 | 0 |
| expmant+huf | 15.80 | 22.2 | 0 | 669 | 1549 | 186.5 | 3 | 4.27 5.18 5.70 | 0 |
| bytes+huf | 13.74 | 25.5 | 0 | 1110 | 3009 | 33.9 | 3 | 4.25 5.23 5.72 | 0 |
| expwin-bf16 | 1.00 | 350.6 | 0 | 14574 | 15771 | 0.0 | 3 | 4.59 5.18 5.68 | 0 |
| bytes+raw | 1.00 | 350.6 | 0 | 6127 | 10046 | 33.9 | 3 | 4.19 5.23 5.73 | 0 |

</details>

<details><summary>one-model: 138 tensors, 4174.8 MB, sorted by size</summary>

| method | ratio | out MB | select meta B | enc MB/s/core | dec MB/s/core | dec heap peak MB | reps | load | failures |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| adaptive-per-tensor | 16182 | 0.26 | 138 | 156 | 68474 | 6.6 | 3 | 6.65 5.60 5.59 | 0 |
| sp-runs-dict+zstd19 | 16093 | 0.26 | 0 | 1507 | 71545 | 6.6 | 3 | 6.27 5.40 5.52 | 0 |
| gated-per-tensor | 15824 | 0.26 | 140 | 821 | 84608 | 6.0 | 3 | 4.43 4.42 4.50 | 0 |
| sp-runs-dict+zstd3 | 12364 | 0.34 | 0 | 1765 | 67868 | 6.6 | 3 | 6.27 5.40 5.52 | 0 |
| sp-runs+zstd3 | 11785 | 0.35 | 0 | 1809 | 71226 | 7.0 | 3 | 6.03 5.34 5.50 | 0 |
| sp-csr+zstd3 | 11173 | 0.37 | 0 | 2478 | 57782 | 7.6 | 3 | 6.03 5.34 5.50 | 0 |
| sp-bitmap+zstd3 | 8713 | 0.48 | 0 | 2333 | 22964 | 46.5 | 3 | 5.86 5.28 5.49 | 0 |
| sp-rowdedup-csr-dict+zstd19 | 8186 | 0.51 | 0 | 695 | 31300 | 378.1 | 3 | 6.42 5.47 5.55 | 0 |
| sp-rowdedup-csr-dict+zstd3 | 7056 | 0.59 | 0 | 747 | 31949 | 378.1 | 3 | 6.46 5.47 5.55 | 0 |
| adaptive-per-block | 5867 | 0.71 | 9415 | 172 | 10458 | 5.8 | 3 | 6.69 5.72 5.64 | 0 |
| bytegroup+zstd19 | 5009 | 0.83 | 0 | 322 | 3845 | 1115.1 | 3 | 6.66 5.53 5.72 | 0 |
| bytes+zstd19L | 4548 | 0.92 | 0 | 71 | 12792 | 743.4 | 3 | 4.83 5.17 5.66 | 0 |
| bytes+zstd19 | 4381 | 0.95 | 0 | 346 | 13378 | 743.4 | 3 | 4.27 5.07 5.63 | 0 |
| xor-row+zstd3 | 4320 | 0.97 | 0 | 968 | 1511 | 1115.1 | 3 | 5.01 5.05 5.42 | 0 |
| bytes+zstd9 | 4056 | 1.03 | 0 | 1223 | 10930 | 743.4 | 3 | 4.27 5.07 5.63 | 0 |
| xor-tile+zstd3 | 3889 | 1.07 | 0 | 571 | 519 | 3345.3 | 3 | 5.17 5.08 5.43 | 0 |
| expmant+zstd19 | 3816 | 1.09 | 0 | 225 | 2874 | 2044.4 | 3 | 5.26 5.38 5.65 | 0 |
| bytegroup+zstd3 | 3547 | 1.18 | 0 | 1803 | 2908 | 1115.1 | 3 | 5.24 5.23 5.62 | 0 |
| dedup-rows+zstd3 | 3358 | 1.24 | 0 | 1446 | 8214 | 1481.4 | 3 | 5.76 5.25 5.48 | 0 |
| dedup-4k+zstd3 | 3332 | 1.25 | 0 | 2183 | 32086 | 313.9 | 3 | 5.86 5.28 5.49 | 0 |
| xor-prev+zstd3 | 3284 | 1.27 | 0 | 967 | 706 | 1115.1 | 3 | 4.91 5.03 5.42 | 0 |
| sp-runs-dict+huf | 3092 | 1.35 | 0 | 1767 | 67824 | 6.6 | 3 | 6.41 5.44 5.54 | 0 |
| sp-runs-dict+rans | 2922 | 1.43 | 0 | 1761 | 53873 | 6.6 | 3 | 6.41 5.44 5.54 | 0 |
| bitplane+zstd3 | 2829 | 1.48 | 0 | 404 | 178 | 1115.1 | 3 | 3.92 4.97 5.46 | 0 |
| bytes+zstd3 | 2816 | 1.48 | 0 | 4679 | 8408 | 743.4 | 3 | 4.38 5.11 5.65 | 0 |
| expmant+zstd3 | 2763 | 1.51 | 0 | 1222 | 1934 | 2044.4 | 3 | 5.28 5.38 5.65 | 0 |
| sp-rowdedup-csr-dict+rans | 2728 | 1.53 | 0 | 741 | 28124 | 378.1 | 3 | 6.47 5.50 5.56 | 0 |
| bytes+zstd3-1MiB | 2631 | 1.59 | 0 | 7166 | 17088 | 0.0 | 3 | 5.24 5.23 5.62 | 0 |
| bytes+zstd1 | 2619 | 1.59 | 0 | 6176 | 7853 | 743.4 | 3 | 4.38 5.11 5.65 | 0 |
| sp-bitmap+rans | 2419 | 1.73 | 0 | 1181 | 2662 | 25.6 | 3 | 5.95 5.31 5.50 | 0 |
| bytes+zstd3-1MiB+dict | 2385 | 1.75 | 0 | 7091 | 16804 | 0.0 | 3 | 5.24 5.23 5.62 | 0 |
| delta-lin+zstd3 | 2194 | 1.90 | 0 | 589 | 458 | 1858.5 | 3 | 5.69 5.21 5.47 | 0 |
| delta-prev+zstd3 | 1833 | 2.28 | 0 | 664 | 844 | 1858.5 | 3 | 5.54 5.17 5.46 | 0 |
| expmant+tans | 1324 | 3.15 | 0 | 505 | 688 | 2044.4 | 3 | 5.05 5.32 5.62 | 0 |
| bytes+tans | 1303 | 3.20 | 0 | 628 | 717 | 371.7 | 3 | 4.49 5.04 5.58 | 0 |
| bytegroup+rans | 1216 | 3.43 | 0 | 145 | 184 | 1115.1 | 3 | 6.14 5.49 5.70 | 0 |
| bytes+rans | 1124 | 3.71 | 0 | 149 | 191 | 371.7 | 3 | 5.54 5.25 5.65 | 0 |
| expmant+rans | 920 | 4.54 | 0 | 97 | 123 | 2044.4 | 3 | 4.89 5.28 5.60 | 0 |
| bitplane+rans | 838 | 4.98 | 0 | 112 | 92 | 1115.1 | 3 | 4.35 4.99 5.45 | 0 |
| zfp-reversible(f32) | 227 | 18.4 | 0 | 587 | 859 | 743.4 | 3 | 6.39 5.52 5.56 | 0 |
| bytegroup+huf | 23.12 | 180.6 | 0 | 992 | 2207 | 1115.1 | 3 | 6.63 5.56 5.73 | 0 |
| zipnn-reimpl | 23.12 | 180.6 | 0 | 902 | 3063 | 0.8 | 3 | 6.51 5.52 5.57 | 0 |
| expmant+huf | 15.64 | 267.0 | 0 | 675 | 1552 | 2044.4 | 3 | 5.44 5.40 5.65 | 0 |
| bytes+huf | 12.92 | 323.1 | 0 | 1058 | 2806 | 371.7 | 3 | 4.44 5.04 5.58 | 0 |
| expwin-bf16 | 1.00 | 4174.8 | 0 | 13521 | 13345 | 0.0 | 3 | 6.65 5.60 5.59 | 0 |
| bytes+raw | 1.00 | 4174.8 | 0 | 6355 | 12560 | 371.7 | 3 | 4.41 5.13 5.66 | 0 |

</details>

<details><summary>gemma3-1b-f16: 340 tensors, 2000.0 MB, sorted by size</summary>

| method | ratio | out MB | select meta B | enc MB/s/core | dec MB/s/core | dec heap peak MB | reps | load | failures |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| sp-runs-dict+rans | 1.53 | 1309.2 | 0 | 50 | 87 | 2013.3 | 1 | 3.29 4.16 4.71 | 0 |
| sp-rowdedup-csr-dict+rans | 1.53 | 1311.0 | 0 | 46 | 86 | 2284.4 | 1 | 3.21 3.86 4.42 | 0 |
| adaptive-per-tensor | 1.52 | 1316.3 | 342 | 1 | 379 | 2013.3 | 1 | 3.68 3.91 4.42 | 0 |
| sp-runs-dict+zstd19 | 1.52 | 1316.6 | 0 | 2 | 360 | 2013.3 | 1 | 5.14 4.69 4.98 | 0 |
| bytegroup+rans | 1.52 | 1316.7 | 0 | 158 | 150 | 1811.9 | 3 | 4.38 5.15 5.42 | 0 |
| sp-bitmap+rans | 1.52 | 1316.8 | 0 | 124 | 123 | 1849.7 | 3 | 4.28 4.54 5.04 | 0 |
| sp-rowdedup-csr-dict+zstd19 | 1.52 | 1318.0 | 0 | 2 | 340 | 2284.4 | 1 | 3.57 4.15 4.68 | 0 |
| expmant+rans | 1.51 | 1322.2 | 0 | 105 | 104 | 3321.9 | 3 | 3.39 4.70 5.23 | 0 |
| adaptive-per-block | 1.51 | 1325.0 | 8610 | 3 | 443 | 7.9 | 1 | 8.70 5.82 5.14 | 0 |
| sp-runs-dict+huf | 1.51 | 1326.4 | 0 | 85 | 356 | 2013.3 | 1 | 3.17 4.17 4.72 | 0 |
| expmant+tans | 1.51 | 1327.7 | 0 | 281 | 336 | 3321.9 | 3 | 3.62 4.81 5.27 | 0 |
| bytegroup+zstd3 | 1.50 | 1332.6 | 0 | 691 | 1260 | 1811.9 | 3 | 4.40 5.19 5.44 | 0 |
| sp-bitmap+zstd3 | 1.50 | 1332.6 | 0 | 369 | 523 | 1849.7 | 3 | 4.34 4.56 5.06 | 0 |
| sp-runs+zstd3 | 1.50 | 1332.7 | 0 | 271 | 350 | 6408.9 | 3 | 4.39 4.55 5.03 | 0 |
| sp-csr+zstd3 | 1.50 | 1332.7 | 0 | 266 | 332 | 6411.5 | 3 | 4.23 4.51 5.00 | 0 |
| bytegroup+zstd19 | 1.50 | 1333.8 | 0 | 3 | 1181 | 805.3 | 1 | 2.95 3.66 4.33 | 0 |
| zipnn-reimpl | 1.50 | 1333.8 | 0 | 808 | 1427 | 0.8 | 3 | 4.16 4.49 4.98 | 0 |
| bytegroup+huf | 1.50 | 1333.8 | 0 | 736 | 1274 | 1811.9 | 3 | 4.41 5.17 5.43 | 0 |
| dedup-rows+zstd3 | 1.50 | 1334.3 | 0 | 413 | 1198 | 2418.0 | 3 | 4.63 4.62 5.09 | 0 |
| expmant+huf | 1.49 | 1341.6 | 0 | 492 | 925 | 3321.9 | 3 | 3.66 4.86 5.29 | 0 |
| gated-per-tensor | 1.49 | 1341.6 | 342 | 396 | 958 | 1476.4 | 3 | 4.47 4.43 4.50 | 0 |
| expmant+zstd19 | 1.49 | 1342.9 | 0 | 1 | 902 | 1476.4 | 1 | 3.70 4.10 4.38 | 0 |
| sp-runs-dict+zstd3 | 1.43 | 1395.3 | 0 | 75 | 319 | 2013.3 | 1 | 4.12 4.47 4.90 | 0 |
| sp-rowdedup-csr-dict+zstd3 | 1.43 | 1396.8 | 0 | 67 | 305 | 2284.4 | 1 | 3.75 4.21 4.71 | 0 |
| xor-row+zstd3 | 1.41 | 1414.3 | 0 | 523 | 861 | 1811.9 | 3 | 4.15 4.50 5.07 | 0 |
| xor-prev+zstd3 | 1.41 | 1415.0 | 0 | 537 | 512 | 1811.9 | 3 | 3.92 4.47 5.07 | 0 |
| xor-tile+zstd3 | 1.40 | 1427.0 | 0 | 404 | 425 | 5435.8 | 3 | 4.38 4.54 5.08 | 0 |
| bitplane+rans | 1.38 | 1450.8 | 0 | 121 | 87 | 1811.9 | 3 | 3.16 4.42 5.08 | 0 |
| bitplane+zstd3 | 1.37 | 1455.1 | 0 | 334 | 173 | 1811.9 | 3 | 3.19 4.50 5.12 | 0 |
| expmant+zstd3 | 1.34 | 1491.1 | 0 | 157 | 603 | 3321.9 | 3 | 3.87 4.97 5.34 | 0 |
| bytes+rans | 1.31 | 1521.4 | 0 | 156 | 144 | 604.0 | 3 | 4.46 5.31 5.49 | 0 |
| bytes+tans | 1.31 | 1529.6 | 0 | 483 | 545 | 604.0 | 3 | 4.65 5.37 5.51 | 0 |
| bytes+zstd9 | 1.31 | 1529.8 | 0 | 75 | 963 | 1208.0 | 3 | 5.77 5.61 5.60 | 0 |
| bytes+zstd19L | 1.31 | 1530.4 | 0 | 2 | 949 | 536.9 | 1 | 4.11 4.28 4.75 | 0 |
| bytes+zstd19 | 1.31 | 1530.6 | 0 | 3 | 916 | 536.9 | 1 | 4.19 4.47 4.95 | 0 |
| bytes+huf | 1.30 | 1534.5 | 0 | 936 | 1415 | 604.0 | 3 | 4.65 5.37 5.51 | 0 |
| bytes+zstd3 | 1.30 | 1534.5 | 0 | 1022 | 1170 | 1208.0 | 3 | 5.77 5.61 5.60 | 0 |
| bytes+zstd3-1MiB | 1.30 | 1534.5 | 0 | 1071 | 1213 | 0.0 | 3 | 4.43 5.21 5.45 | 0 |
| bytes+zstd1 | 1.30 | 1534.5 | 0 | 1148 | 1174 | 1208.0 | 3 | 6.19 5.69 5.63 | 0 |
| bytes+zstd3-1MiB+dict | 1.30 | 1534.7 | 0 | 907 | 1194 | 0.0 | 3 | 4.05 4.34 4.54 | 0 |
| dedup-4k+zstd3 | 1.30 | 1535.6 | 0 | 498 | 1168 | 1208.5 | 3 | 4.42 4.58 5.07 | 0 |
| delta-prev+zstd3 | 1.24 | 1608.8 | 0 | 436 | 511 | 3019.9 | 3 | 4.86 4.65 5.11 | 0 |
| delta-lin+zstd3 | 1.09 | 1833.6 | 0 | 350 | 347 | 3019.9 | 3 | 5.02 4.69 5.12 | 0 |
| expwin-bf16 | 1.00 | 2000.0 | 0 | 17725 | 22587 | 0.0 | 3 | 4.29 4.50 4.96 | 0 |
| bytes+raw | 1.00 | 2000.0 | 0 | 7788 | 12208 | 604.0 | 3 | 6.46 5.73 5.64 | 0 |
| zfp-reversible(f32) | 0.98 | 2032.1 | 0 | 134 | 133 | 1208.0 | 3 | 4.06 4.46 4.97 | 0 |

</details>

<details><summary>gemma4-e4b-bf16-blk20: 17 tensors, 186.2 MB, sorted by size</summary>

| method | ratio | out MB | select meta B | enc MB/s/core | dec MB/s/core | dec heap peak MB | reps | load | failures |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| sp-runs-dict+rans | 1.51 | 123.4 | 0 | 50 | 89 | 452.0 | 1 | 4.21 4.35 4.68 | 0 |
| sp-rowdedup-csr-dict+rans | 1.51 | 123.5 | 0 | 47 | 86 | 504.7 | 1 | 4.77 4.49 4.72 | 0 |
| expmant+rans | 1.50 | 123.7 | 0 | 154 | 155 | 262.1 | 3 | 3.47 4.98 5.00 | 0 |
| expmant+tans | 1.50 | 124.0 | 0 | 628 | 908 | 262.1 | 3 | 3.47 4.98 5.00 | 0 |
| sp-runs-dict+huf | 1.50 | 124.1 | 0 | 90 | 363 | 452.0 | 1 | 4.21 4.35 4.68 | 0 |
| adaptive-per-tensor | 1.50 | 124.2 | 17 | 1 | 416 | 452.0 | 1 | 4.63 4.46 4.71 | 0 |
| sp-runs-dict+zstd19 | 1.50 | 124.2 | 0 | 2 | 364 | 452.0 | 1 | 3.87 4.31 4.69 | 0 |
| sp-rowdedup-csr-dict+zstd19 | 1.50 | 124.3 | 0 | 2 | 332 | 504.7 | 1 | 4.11 4.33 4.67 | 0 |
| adaptive-per-block | 1.50 | 124.4 | 757 | 3 | 2206 | 5.2 | 1 | 3.91 4.29 4.61 | 0 |
| zipnn-reimpl | 1.50 | 124.4 | 0 | 1025 | 2082 | 0.8 | 3 | 3.48 4.84 4.95 | 0 |
| expmant+huf | 1.50 | 124.4 | 0 | 887 | 2178 | 262.1 | 3 | 3.51 5.01 5.01 | 0 |
| gated-per-tensor | 1.50 | 124.4 | 17 | 661 | 2071 | 262.1 | 3 | 3.72 4.84 4.95 | 0 |
| expmant+zstd19 | 1.49 | 124.6 | 0 | 2 | 1832 | 262.1 | 1 | 3.64 4.36 4.73 | 0 |
| bytegroup+rans | 1.49 | 124.8 | 0 | 155 | 155 | 157.3 | 3 | 3.38 5.01 5.01 | 0 |
| sp-bitmap+rans | 1.49 | 124.8 | 0 | 124 | 126 | 160.6 | 3 | 3.38 4.87 4.96 | 0 |
| bytegroup+huf | 1.48 | 125.4 | 0 | 941 | 2341 | 157.3 | 3 | 3.38 5.01 5.01 | 0 |
| bytegroup+zstd19 | 1.48 | 125.7 | 0 | 2 | 1942 | 157.3 | 1 | 3.54 4.46 4.79 | 0 |
| expwin-bf16 | 1.42 | 131.2 | 0 | 157 | 1217 | 89.4 | 3 | 3.53 4.82 4.94 | 0 |
| sp-runs-dict+zstd3 | 1.41 | 131.7 | 0 | 74 | 341 | 452.0 | 1 | 3.85 4.31 4.69 | 0 |
| sp-rowdedup-csr-dict+zstd3 | 1.41 | 131.7 | 0 | 69 | 310 | 504.7 | 1 | 4.11 4.33 4.67 | 0 |
| expmant+zstd3 | 1.41 | 132.0 | 0 | 338 | 1359 | 262.1 | 3 | 3.51 5.01 5.01 | 0 |
| bytegroup+zstd3 | 1.40 | 133.1 | 0 | 325 | 1312 | 157.3 | 3 | 3.38 5.01 5.01 | 0 |
| sp-bitmap+zstd3 | 1.40 | 133.1 | 0 | 235 | 545 | 160.6 | 3 | 3.38 4.87 4.96 | 0 |
| sp-runs+zstd3 | 1.40 | 133.1 | 0 | 190 | 349 | 451.9 | 3 | 3.35 4.83 4.95 | 0 |
| sp-csr+zstd3 | 1.40 | 133.1 | 0 | 183 | 330 | 452.1 | 3 | 3.48 4.84 4.95 | 0 |
| dedup-rows+zstd3 | 1.40 | 133.2 | 0 | 242 | 1231 | 209.8 | 3 | 3.50 4.91 4.98 | 0 |
| xor-prev+zstd3 | 1.38 | 135.2 | 0 | 264 | 557 | 157.3 | 3 | 3.55 4.95 4.99 | 0 |
| xor-row+zstd3 | 1.36 | 137.3 | 0 | 252 | 955 | 157.3 | 3 | 3.55 4.95 4.99 | 0 |
| bitplane+rans | 1.36 | 137.4 | 0 | 121 | 89 | 157.3 | 3 | 3.51 4.96 5.00 | 0 |
| xor-tile+zstd3 | 1.35 | 137.6 | 0 | 219 | 480 | 471.9 | 3 | 3.55 4.95 4.99 | 0 |
| bitplane+zstd3 | 1.35 | 137.8 | 0 | 338 | 175 | 157.3 | 3 | 3.51 4.96 5.00 | 0 |
| delta-prev+zstd3 | 1.31 | 141.6 | 0 | 251 | 674 | 262.1 | 3 | 3.50 4.91 4.98 | 0 |
| bytes+zstd19 | 1.29 | 144.2 | 0 | 3 | 615 | 104.9 | 1 | 3.59 4.79 4.93 | 0 |
| bytes+zstd19L | 1.29 | 144.2 | 0 | 2 | 620 | 104.9 | 1 | 3.33 4.59 4.85 | 0 |
| bytes+zstd9 | 1.29 | 144.7 | 0 | 80 | 885 | 104.9 | 3 | 3.63 5.12 5.05 | 0 |
| bytes+rans | 1.28 | 145.4 | 0 | 160 | 150 | 52.5 | 3 | 3.50 5.07 5.03 | 0 |
| bytes+tans | 1.28 | 145.7 | 0 | 473 | 550 | 52.4 | 3 | 3.50 5.07 5.03 | 0 |
| bytes+huf | 1.28 | 145.8 | 0 | 918 | 1332 | 52.4 | 3 | 3.50 5.07 5.03 | 0 |
| bytes+zstd1 | 1.28 | 145.9 | 0 | 1064 | 1107 | 104.9 | 3 | 3.63 5.12 5.05 | 0 |
| bytes+zstd3-1MiB | 1.28 | 145.9 | 0 | 966 | 1125 | 0.0 | 3 | 3.38 5.01 5.01 | 0 |
| bytes+zstd3 | 1.28 | 145.9 | 0 | 920 | 1087 | 104.9 | 3 | 3.63 5.12 5.05 | 0 |
| bytes+zstd3-1MiB+dict | 1.28 | 146.0 | 0 | 801 | 1098 | 0.0 | 3 | 4.27 4.38 4.56 | 0 |
| dedup-4k+zstd3 | 1.28 | 146.0 | 0 | 529 | 1096 | 104.9 | 3 | 3.38 4.87 4.96 | 0 |
| delta-lin+zstd3 | 1.20 | 155.2 | 0 | 277 | 416 | 262.1 | 3 | 3.50 4.91 4.98 | 0 |
| bytes+raw | 1.00 | 186.2 | 0 | 7764 | 15640 | 52.4 | 3 | 3.63 5.12 5.05 | 0 |
| zfp-reversible(f32) | 0.97 | 191.4 | 0 | 156 | 147 | 104.9 | 3 | 3.48 4.84 4.95 | 0 |

</details>

<details><summary>gemma4-e4b-bf16-full: 746 tensors, 15037.3 MB, sorted by size</summary>

| method | ratio | out MB | select meta B | enc MB/s/core | dec MB/s/core | dec heap peak MB | reps | load | failures |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| expmant+tans | 1.51 | 9958.1 | 0 | 596 | 874 | 1342.2 | 3 | 4.13 4.31 4.51 | 0 |
| zipnn-reimpl | 1.50 | 9995.2 | 0 | 1003 | 2037 | 0.8 | 3 | 4.10 4.29 4.48 | 0 |
| expmant+huf | 1.50 | 9995.2 | 0 | 881 | 1936 | 1342.2 | 3 | 4.38 4.35 4.53 | 0 |
| gated-per-tensor | 1.50 | 9995.2 | 746 | 639 | 1993 | 1342.2 | 3 | 3.64 4.28 4.46 | 0 |
| bytegroup+huf | 1.49 | 10095.7 | 0 | 917 | 2051 | 805.3 | 3 | 4.27 4.33 4.53 | 0 |
| expwin-bf16 | 1.42 | 10578.7 | 0 | 158 | 1253 | 457.3 | 3 | 4.04 4.27 4.47 | 0 |
| bytegroup+zstd3 | 1.40 | 10716.3 | 0 | 324 | 1344 | 805.3 | 3 | 3.93 4.18 4.50 | 0 |
| bytes+huf | 1.28 | 11748.3 | 0 | 879 | 1326 | 268.4 | 3 | 3.99 4.20 4.52 | 0 |
| bytes+zstd1 | 1.28 | 11748.5 | 0 | 1005 | 1083 | 536.9 | 3 | 3.78 4.23 4.58 | 0 |
| bytes+zstd3 | 1.28 | 11749.8 | 0 | 891 | 1082 | 536.9 | 3 | 3.63 4.15 4.53 | 0 |

</details>

### What the GBC results say

Sparsity decides everything on the compiled weights. The best general-purpose coder, `bytes+zstd19L`, reaches 4,548 times. Any form that drops the zeros before entropy coding does 2 to 4 times better again. `sp-runs-dict+zstd19` codes the one-model's f16 matrices in 257,249 bytes, 1.39 bits per nonzero. That includes each nonzero's position and value. Positions drawn at random at this density would cost about 12 bits each (computed as log2 of 1,405 elements per nonzero, plus 1.44), so zstd finds repeated structure in the run lengths. Row deduplication does not help. `sp-rowdedup-csr-dict+zstd19` gives 8,186 times against 16,093. The cause was not isolated. A likely one is that CSR restarts its column gaps on every row, which breaks up the long runs that the single zero-run stream keeps (a guess).

Huffman cannot go below 1 bit per byte, so any method that ends in huff0 stops between 13 and 23 times on this data. This includes ZipNN, `bytegroup+huf` and `expmant+huf`. Only huff0's whole-chunk RLE case gets past that. tANS and rANS go below 1 bit but do not see runs, so they land between 300 and 1,300 times. LZ coders (zstd) see runs, and the sparse forms remove them, so those two families lead.

ZFP reversible gets 227 times on the one-model because a block of 4×4 zeros codes small. It still pays the 2 times widening, and it stays far below zstd -1 (2,619 times).

### What the trained-model results say

On trained weights every sensible method lands between 1.48 and 1.53 times, and the spread is speed. The exponent byte holds the redundancy, so the methods that separate it (`bytegroup`, `expmant`, ZipNN) reach 1.50. Plain zstd on bytes reaches 1.28 to 1.31. Coding with tANS or rANS instead of Huffman adds 1 to 2% and decodes 3 to 16 times slower. XOR and delta transforms lose to plain byte grouping, which suggests that neighbouring mantissas in a trained matrix are not correlated (inferred, not measured). Trained zstd dictionaries add nothing on any set. On the GBC models the trainer produced a 527-byte dictionary that made 1 MiB chunks larger, not smaller.

## Whole-file load time

`load.sh` packs each GGUF into the container with `gated-per-tensor`, rebuilds it 3 times into a new file and 3 times into memory, and byte-compares the rebuilt file with the original. Two floors give the limiter. The fill floor below is a single run, so a rebuild that lands a few milliseconds under it is noise, not a decoder beating a memset. The file floor is `cp` of the original into a new file, which reads from the page cache and writes a new file. The memory floor allocates the same number of bytes and writes every one of them on 4 threads (`weight-compress fill`). Peak RSS comes from `/usr/bin/time -l`. For a rebuild into a file, RSS includes the dirty pages of the mapped output file. For a rebuild into memory, RSS includes the output buffer.

Measured, 4 threads, 3 runs each, median with the range in brackets. The load average was 3.7 to 5.1 during these runs.

| model | GGUF bytes | container bytes | ratio | pack | rebuild into a file, before `msync` | with `msync` | `cp` floor | rebuild into memory | fill floor | rebuild peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu-memory | 350,814,112 | 75,189 | 4,666 | 0.24 s | 0.040 s [0.033 to 0.041] | 0.30 to 0.34 s | 0.15 s [0.13 to 0.17] | 0.013 s [0.012 to 0.013] | 0.011 s | 360 MB |
| one-model | 4,175,789,856 | 274,046 | 15,238 | 1.58 s | 0.378 s [0.349 to 0.493] | 1.33 to 3.06 s | 1.51 s [1.49 to 1.85] | 0.142 s [0.141 to 0.142] | 0.164 s | 4.19 GB |
| Gemma 3 1B f16 | 2,006,573,728 | 1,343,003,379 | 1.494 | 3.94 s | 1.025 s [1.023 to 1.054] | 1.47 to 1.50 s | 0.70 s [0.69 to 0.70] | 0.987 s [0.970 to 1.013] | 0.050 s | 5.5 GB |
| Gemma 4 E4B bf16 | 15,053,097,408 | 9,998,541,434 | 1.506 | 46.1 s | 13.2 s [9.5 to 17.3] | 10.9 to 20.5 s | 6.05 s [5.97 to 6.12] | 4.79 s [4.75 to 19.2] | 1.14 s | 27.6 GB |

The same files through the zstd CLI (`zstd -T4` to encode, one thread to decode, decode time the mean of 3):

| model | `zstd -3` bytes | decode | `zstd -19 --long=31` bytes | encode | decode |
|---|---:|---:|---:|---:|---:|
| cpu-memory | 278,607 | 0.17 s | 154,655 | 2.8 s | 0.15 s |
| one-model | 2,078,522 | 1.36 s | 937,635 | 45.0 s | 0.89 s |
| Gemma 3 1B f16 | 1,536,926,377 | 1.99 s | 1,508,850,917 | 373.7 s | 3.76 s |
| Gemma 4 E4B bf16 | 11,755,049,566 | 16.8 s | not run | | |

`zstd -19` was not run on Gemma 4. At the 2 to 3 MB/s per core measured on Gemma 3, it would take about half an hour, and the layer-20 rows already give its ratio (1.29).

The container is 3.4 times smaller than `zstd -19 --long=31` on the one-model and 2.1 times smaller on cpu-memory. It rebuilds the one-model GGUF 2.4 times faster than zstd decodes it (0.38 s against 0.89 s), using 4 threads against zstd's one.

Gemma 4's rebuild varies by a factor of 2 to 4 between runs, and its first run into memory is 4 times slower than the other two. The rebuild holds the 10 GB container mapping and the 15 GB output at once (27.6 GB peak RSS), while other jobs held about 13 GB of the machine's 64 GB. So the slow runs are bound by memory pressure, not by the decoder (inferred from the RSS and the spread, not profiled). The two unpressured runs into memory reach 3.1 GB/s on 4 threads.

The GBC rebuild is bound by writing the output. Into memory it takes 0.14 s for 4.18 GB, against 0.16 s to allocate and write the same bytes, so the decoder costs nothing measurable beyond the write. Into a file it takes 0.35 to 0.49 s before `msync`, faster than `cp` at 1.49 to 1.85 s, because nothing has to be read. Gemma 3 is bound by huff0 decoding at about 2 GB/s across 4 threads, about 1.4 times the `cp` floor.

The pass latency after loading is unchanged by construction: the rebuilt GGUF is byte-identical to the original, so llama.cpp loads the same bytes. `tools/stock-llama/bench.sh` was not rerun, since no method changes what llama.cpp loads.

## GPU decode

Two Metal compute kernels decode straight into a GPU buffer (`src/gpu.rs`, run by `gpu.sh` while it holds the shared GPU lock). Both buffers use `StorageModeShared`, which on Apple silicon is the same memory the GPU reads. The harness fills each output buffer with `0xA5` before every timed run, so the kernel must write every byte, and it byte-compares every tensor after every run. GPU times come from the command buffer's `GPUStartTime` and `GPUEndTime`. Load average was 2.8 to 4.1.

**The sparse expand, for the GBC checkpoints.** The format holds, per block of 32,768 elements, a u32 entry per nonzero (`dict index << 15 | position in block`) and per tensor a table of distinct values. One threadgroup of 256 threads owns a block. It writes the block's zeros in 16-byte stores, waits on a barrier, then scatters its entries. The output buffer is the GGUF's data section, so each tensor lands at its own offset.

| | cpu-memory | one-model |
|---|---:|---:|
| decoded tensor bytes | 350,601,216 | 4,174,848,000 |
| GPU-resident compressed form | 1,379,296 B (254 times) | 8,624,324 B (484 times) |
| the same form on disk (zstd -19) | 84,345 B; Mac CPU decode 0.71 ms | 320,579 B; Mac CPU decode 2.56 ms |
| VRAM, measured (`currentAllocatedSize`) | 1,392,640 B compressed + 350,601,216 B output | 8,634,368 B compressed + 4,174,856,192 B output |
| GPU, whole model in one dispatch (median of 5) | 1.30 ms [1.19 to 1.93], 270 GB/s | 12.25 ms [11.24 to 12.75], 341 GB/s |
| GPU, one dispatch per tensor (1 run) | median 28.8 µs, sum 4.0 ms | median 40.8 µs, sum 19.0 ms |
| Mac CPU, same format into the shared buffer, 4 threads (median of 5) | 1.61 ms, 218 GB/s | 17.4 ms, 240 GB/s |
| Mac CPU decode to host memory, then `memcpy` into the shared buffer | +4.29 ms (82 GB/s) | +49.6 ms (84 GB/s) |
| GPU blit, shared to private buffer | 2.06 ms (171 GB/s) | 23.1 ms (180 GB/s) |
| tensor mismatches, all runs | 0 | 0 |

The GPU expand of the one-model runs at 341 GB/s against the M1 Max's 400 GB/s peak memory bandwidth, so it is bound by writing the output, and the scatter of 1.6M entries does not show. Per-tensor dispatches add about 7 ms of launch overhead across 138 tensors, so one dispatch per model or per layer is the right shape. The Mac CPU expand writes into memory that the `0xA5` fill had just made resident, so its 240 GB/s is the rate for pages that are already mapped. A fresh allocation adds the page faults, as in the 0.14 s container rebuild. On unified memory there is no copy to make. A discrete-style flow that decodes on the Mac CPU and then copies to a private buffer pays the blit, 23 ms here, on top of the decode.

**`expwin-bf16`, for trained bf16 weights.** Huffman streams do not decode in parallel inside a tensor, so the dense format trades a little size for independence. Each tensor gets a code width k of 2, 3 or 4 bits, chosen for the smallest output. Each block of 1,024 elements gets a base exponent. Each element stores a k-bit code and one byte of sign and mantissa. A code below `2^k - 1` means exponent base + code, and the escape code reads the next byte of the block's escape list. One thread decodes one element. It finds its place in the escape list with `simd_prefix_exclusive_sum` and a 32-entry threadgroup scan. Every Gemma 4 tensor chose k = 3, with 1.9 to 9.6% escapes.

| Gemma 4 E4B, 29 bf16 tensors (layers 0, 20 and 41, `token_embd`, `per_layer_model_proj`), 1.98 GB | |
|---|---:|
| ratio | 1.422 (against 1.50 for `expmant+huf` on the same file) |
| GPU, one dispatch per tensor (median of 5 each) | sum 38.3 ms, 51.8 GB/s; 18 to 70 GB/s per tensor |
| Mac CPU, same format, 4 threads | sum 373 ms, 5.3 GB/s |
| VRAM per tensor, computed | compressed form plus output, for example 0.94 GB + 1.34 GB for `token_embd` |
| tensor mismatches | 0 |

The expwin kernel reaches 18 to 70 GB/s, far below the bandwidth limit. Tensors of the same size split into two groups near 29 and 55 GB/s. The limiter was not profiled. The likely causes are one 2-byte store per thread and two barriers per 1,024-thread group, so the next step would be 8 elements per thread with one 16-byte store (a guess). The `currentAllocatedSize` reading in this run (2.3 MB peak) cannot be right for a 1.34 GB output buffer, so the VRAM row above is computed instead.

**Decoding on the fly, every pass (computed).** A llama.cpp pass over the one-model's 1,024 cells takes 13.6 ms at 8 tokens and 94 ms at 185 tokens, and most of that is matmul. Expanding the weights to dense every pass would add the measured 12.25 ms, an extra 90% at 8 tokens and 13% at 185, and the matmul would still read the full 4.17 GB it just wrote. So dense expansion per pass does not pay. Weights that stay compressed on the GPU only pay off when the kernel reads the 8.6 MB of entries directly, as a sparse matmul would (not built here). For trained bf16 weights the same holds: expanding 15 GB of Gemma 4 at 52 GB/s would take about 290 ms per pass. The format can only pay inside a fused matmul that reads 1.42 times fewer weight bytes, so a bandwidth-bound pass could get up to 1.42 times faster (a guess, not measured).

**Load-time decompression** is the path that works today. The container rebuilds the GGUF, the bytes are identical, so llama.cpp's load and pass latency do not change (see [Whole-file load time](#whole-file-load-time)).

## Recommendation

Use `gated-per-tensor` in the whole-file container (`weight-compress pack <gguf> <out.wcz> --method gated-per-tensor`).

- **Size.** Its output is at most 2.3% larger than the exhaustive per-tensor search on every set (2.3% on one-model, 1.9% on Gemma 3, 0.2% on Gemma 4 layer 20, 0.5% on cpu-memory). The search takes 5 to 660 times longer to encode. On the one-model it is 3.5 times smaller than the best plain zstd (`bytes+zstd19L`), 2.2 times on cpu-memory, and 680 times smaller than ZipNN.
- **Decode cost.** On sparse tensors the decoder writes zeros and scatters about 1.6M values, and the measured rebuild is no slower than allocating and writing the output (0.14 s against 0.16 s for 4.18 GB). On dense tensors it is huff0 at 1.9 to 2.1 GB/s per core on bf16, the same as the ZipNN reimplementation.
- **Simplicity.** One nonzero count decides the codec, the metadata is one byte per tensor, and the decoder needs zstd and huff0.

Two smaller choices remain open. On f16, `bytegroup+huf` is 0.7% smaller than `expmant+huf` (1.50 against 1.49 on Gemma 3), so the dense path could follow ZipNN's per-dtype choice. If decode speed does not matter, `sp-runs-dict+rans` and `expmant+tans` gain another 1 to 2% on trained weights.

For the GPU, keep the sparse form resident as entries and expand per layer or, better, read the entries directly in a sparse kernel. For trained bf16 weights that must stay compressed on the GPU, `expwin-bf16` keeps 95% of the entropy-coded ratio (1.42 against 1.50) and decodes with one thread per element.

## Reproduce

Build the C baselines, then the crate:

```sh
cd tools/compress
mkdir -p vendor && cd vendor
git clone https://github.com/Cyan4973/FiniteStateEntropy.git fse      # 9f30e09
git clone https://github.com/LLNL/zfp.git                              # 2da5c4e
git clone https://github.com/zipnn/zipnn.git                           # 6009704, read only
(cd zfp && mkdir -p build && cd build && cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_UTILITIES=OFF -DBUILD_EXAMPLES=OFF && make -j3)
cd .. && CARGO_BUILD_JOBS=3 cargo build --release
```

`csrc/fse_prefix.h` was generated from `nm -g --defined-only` of the built `libfse.a`. Regenerate it if you move to a different FiniteStateEntropy commit.

Run the benchmarks, the load-time check and the Metal runs:

```sh
tools/compress/bench-all.sh OUT synth cpu-memory one-model gemma3 gemma4-blk20 gemma4-full gated
tools/compress/load.sh WORK gated-per-tensor <gguf> [<gguf> ...]
tools/compress/gpu.sh OUT                      # takes /tmp/gbcformer-gpu.lock while it runs
bun tools/compress/report.ts OUT/*.tsv          # the tables on this page
```

`STOCK_BUILD`, `GEMMA3` and `GEMMA4` override the input paths. Other subcommands:
- `weight-compress inventory <gguf>` prints per-tensor nonzeros, distinct values and unique rows.
- `weight-compress methods` lists the method names.
- `pack`, `unpack` and `verify` build a container, rebuild a GGUF (`-` rebuilds into memory) and byte-compare two files.
- `gpu <gguf>` runs the sparse Metal expand on a whole model, and `gpu-dense <gguf> --prefix p,q` runs the bf16 `expwin` kernel.
