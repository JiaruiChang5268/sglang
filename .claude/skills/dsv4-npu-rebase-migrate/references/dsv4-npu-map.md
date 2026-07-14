# DSV4-NPU compress-state scheduling: where things live & how they flow

The mainline keeps refactoring this, so treat exact line numbers as stale and
**re-locate with `git grep` each time**. The *roles* below are stable even when
files/functions move.

## File-by-file roles

| File | Role |
|---|---|
| `python/sglang/srt/hardware_backend/npu/dsv4_allocator.py` | `DSV4NPUTokenToKVPoolAllocator`. Owns the c4/c128 KV + **state** paged allocators. After the refactor it also owns `compute_dsv4_state_lens_{extend,decode}` (per-req tail-only alloc lens, mutates `req.c{4,128}_state_kv_len` / `_state_alloc_offset` via getattr/setattr) and `_pack_state_lens`. `alloc_extend/alloc_decode` → `_alloc_c_and_state` → `_alloc_state_extend`. NOTE: it does **not** hold a `req_to_token_pool` or `page_size` ref usable at compute time. |
| `python/sglang/srt/hardware_backend/npu/dsv4_common_hooks.py` | Post-alloc hooks `maybe_write_dsv4_{extend,decode}` (scatter the `DSV4OutCacheLoc` bundle into `req_to_token_c{4,128}[_state]`), plus eviction: `maybe_evict_dsv4_state(batch, req, pre_len)`, `maybe_evict_dsv4_state_on_swa`, `_free_state_range`. Eviction takes `batch` (needs `batch.req_to_token_pool`, `batch.tree_cache.page_size`, `batch.token_to_kv_pool_allocator`). |
| `python/sglang/srt/mem_cache/common.py` | Platform-agnostic alloc flow. `_compute_dsv4_state_lens(batch, is_decode=...)` triggers the allocator's compute **right before** the paged alloc (`alloc_paged_token_slots_{extend,decode}`). The natural place to drive **pre-alloc eviction** since `batch` is in scope here. |
| `python/sglang/srt/managers/schedule_batch.py` | After the "move out" refactor this is *thin*: lazily-imported hook calls at extend/decode/evict sites. `maybe_evict_swa` calls `maybe_evict_dsv4_state(self, req, req.seqlen - 1)` post-forward (decode). `Req` has **no** DSV4 field declarations (getattr/setattr). `batch.prefix_lens` (list) and `batch.seq_lens_cpu` are set in `prepare_for_extend`. |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` | The Ascend attention backend. `_build_npu_compress_metadata_prefill` builds `positions_cmp_padding_c{4,128}`, `start_pos`, `seqused`, `c{ratio}_loc`, and zeroes pre-tail `c{ratio}_state_page_table` columns. `forward_compress` calls the fused AscendC op `torch.ops.custom.compressor` (cache_mode=1). The Python per-request compressor loop is **gone** — chunked state handoff happens inside the op via `state_block_table + start_pos`. |
| `python/sglang/srt/hardware_backend/npu/dsv4_req_to_token_pool.py` | `req_to_token_c{4,128}` and `req_to_token_c{4,128}_state` per-req tables (int32; 0 = skip-sentinel). |
| `python/sglang/srt/hardware_backend/npu/dsv4_state_pool.py` / `dsv4_memory_pool.py` | `NPUCompressStatePool` (paged, tail-only); `npu_state_pool_size = max(2, ceil(1.8*ratio/page)+1)*max_num_reqs*page`. `DSV4NPUTokenToKVPool.translate_kv_loc_to_compress_state_loc` **raises** (ring-hash invalid under paged cache_mode=1 — callers must use the bundle / page table). |

## Data flow per alloc step (extend)

```
prepare_for_extend (schedule_batch): set batch.prefix_lens, seq_lens_cpu
  -> alloc_for_extend -> alloc_paged_token_slots_extend (common.py)
      -> _compute_dsv4_state_lens(batch, is_decode=False)   # BEFORE the paged alloc
          -> allocator.compute_dsv4_state_lens_extend(reqs, seq_lens, [prefix_lens])
              # mutates req.c{ratio}_state_kv_len / _alloc_offset / _write_offset
      -> allocator.alloc_extend(...)  -> DSV4OutCacheLoc bundle (stashed on batch.out_cache_loc_dsv4)
  -> maybe_write_dsv4_extend(batch, ...)   # scatter bundle slots into req_to_token_c*_state
      # uses per-req write offset [c{ratio}_state_write_offset, seq) for the state tables
  -> forward -> ascend_dsv4_backend.forward_compress -> torch.ops.custom.compressor
```

Decode is the same shape via `compute_dsv4_state_lens_decode` (+1 slot/req) and
`maybe_write_dsv4_decode` (per-position). Post-forward, `maybe_evict_swa` runs the
decode-cadence eviction.

## Chunked-prefill changes (what the feature adds)

1. **compressor metadata** (`_build_npu_compress_metadata_prefill`):
   - `positions_cmp` = global block starts `arange(prefix//r, total//r) * r` (not
     `req_positions[:cutoff:ratio]`), where `total = prefix + chunk_len`.
   - `start_pos = forward_batch.extend_prefix_lens` (not `zeros`) so the op reads
     the prior-chunk partial-block state and aligns to the global grid.
   - state-page-table tail offset uses **cumulative** seqlen `prefix + chunk_len`.
   - `c{ratio}_loc` = bundle `out_c{4,128}_loc` is the per-chunk increment -> valid
     for chunked (comment was "invalid under chunked").
2. **state alloc lens** (`compute_dsv4_state_lens_extend`, add `prefix_lens`):
   - `prefix_len > 0` (follow-up chunk): allocate only the **incremental** tail
     `count = min(c_alloc_len, chunk_len)`; set `req.c{ratio}_state_write_offset =
     seq - count`; do **not** reset `c{ratio}_state_alloc_offset` (it's the
     eviction low-water mark).
   - `prefix_len == 0`: full tail, seed `alloc_offset = seq - c_alloc_len`
     (bit-exact with upstream); `write_offset == alloc_offset`.
3. **pre-alloc eviction** (`mem_cache/common.py::_compute_dsv4_state_lens`):
   - decode: `maybe_evict_dsv4_state(batch, req, req.seqlen - 1)` for every req
     **before** the decode alloc (fixes the first-decode-step-after-prefill OOM;
     the post-forward `maybe_evict_swa` runs too late and is decode-only).
   - extend: `maybe_evict_dsv4_state(batch, req, prefix_len)` for `prefix_len > 0`
     (frees old pages so the small pool doesn't grow ~c_alloc_len per chunk).
   - it's page-aligned + idempotent, so the existing post-forward call no-ops.
4. **state-table write offset** (`maybe_write_dsv4_extend`): use
   `getattr(r, "c{ratio}_state_write_offset", getattr(r, "c{ratio}_state_alloc_offset", 0))`.

## Why the OOM happens (so you fix the right thing)

The tail-only state pool is ~2 pages (c4=256) / ~3 pages (c128=384) per req, sized
for the *steady-state window*, on the assumption that prefill is a single extend.
Chunked prefill runs the extend `compute` per chunk; the upstream version
**accumulates** `c_alloc_len` (c4's `tail+128` rule = +128 every 128-aligned chunk)
and **resets** `alloc_offset` each chunk, and eviction only runs post-forward in
decode. So the pool fills after ~2 chunks (prefill OOM), and even once that's
bounded, the prefill->decode boundary OOMs because decode eviction is post-alloc.
The fix = incremental tail + pre-alloc eviction (above).

## Validation (token parity, greedy)

- `chunked_prefill_size % page_size == 0` is enforced; under DP-attention it's
  divided by `dp_size` first. So legal values are multiples of `page_size*dp_size`,
  and **per-DP chunk boundaries are always 128-aligned**.
- Consequence: **c128 cross-chunk gather never fires** (no c128 block straddles a
  128-aligned boundary). **c4 overlap is the only cross-chunk path** (it reads the
  previous 4-block across the boundary) — and it fires on every boundary, so a
  chunked-vs-baseline token-parity PASS already covers it.
- Smallest legal chunk for the most c4 boundary stress: `--chunked-prefill-size =
  page_size * dp_size` (per-DP = 128).
- Parity = same prompts, greedy (`temperature 0`), compare **output token ids**
  between chunked and `--chunked-prefill-size -1`. Identical => migration correct.
  Establish a noise floor first if the system isn't fully deterministic
  (`HCCL_DETERMINISTIC=true` helps).
- Helper scripts (if present): `benchmark/dsv4_chunked_parity.py` (synthetic
  prompts) and `benchmark/dsv4_chunked_dataset_test.py` (real dataset + accuracy).

## Quick re-location commands

```bash
git grep -n 'def compute_dsv4_state_lens_extend'                 # allocator
git grep -n 'def maybe_evict_dsv4_state\|def maybe_write_dsv4'   # hooks
git grep -n 'def _compute_dsv4_state_lens'                       # common.py orchestration
git grep -n 'def _build_npu_compress_metadata_prefill'          # backend metadata
git grep -n 'start_pos\|positions_cmp_padding'                   # compressor op metadata
```
