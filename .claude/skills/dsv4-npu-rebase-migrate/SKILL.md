---
name: dsv4-npu-rebase-migrate
description: Rebase or merge a DSV4-NPU feature branch (chunked-prefill, compress-state pool, compressor/allocator changes) onto a heavily-refactored mainline by MIGRATING your changes to where the mainline moved them, not by naively picking a conflict side. Use this whenever a `git rebase`/`merge` of an NPU/DSV4 branch has conflicts in dsv4_allocator.py, dsv4_common_hooks.py, dsv4_schedule_hooks.py, ascend_dsv4_backend.py, schedule_batch.py, or mem_cache/common.py — especially when the mainline "moved/folded/refactored" DSV4 state scheduling out of schedule_batch into the allocator/hooks. Also use when asked to "analyze what migration work is needed" before touching conflicts, or to map feature-branch edits onto a restructured mainline.
---

# Rebasing a DSV4-NPU feature branch onto a refactored mainline

DSV4-NPU code (compress-state pools, c4/c128 compressor, paged allocator, per-req
schedule bookkeeping) gets refactored aggressively: functions get **moved between
files**, **folded into the allocator**, or **deleted for getattr/setattr
conventions**. So rebase conflicts here are usually **"the mainline relocated the
code your commit modified"**, not two edits to the same lines.

**Core move:** a conflict whose HEAD side is empty/relocated means *"this moved."*
Take the mainline side, then re-apply your intent at the code's new home (often a
different file). Keep the non-feature path (`prefix_len == 0`, non-decode)
bit-exact so the common path is provably unchanged.

## Keep it cheap (this is the point)

The expensive way is re-deriving everything with full-file reads and full diffs.
Don't. The concrete map already exists:

1. **Read `references/dsv4-npu-map.md` ONCE.** It has the file roles, data flow,
   the 4 chunked-prefill change sites, the OOM cause, and validation. Don't
   re-derive these by exploring.
2. **Locate with `git grep -n 'def <fn>'`, then read only that function's span** —
   never read a whole 1000-line file to find one function.
3. **Read commit *messages*, not full diffs:** `git show <c> --no-patch --format='%B'`
   tells you what moved where. Only `git show <c> -- <file>` when you must see code.
4. **Batch recon into one command block** (Step 1) instead of many round-trips.
5. **Edit, don't re-read to "verify".** Edit/Write already confirm success; trust them.

## Step 1 — Recon (one batched block)

```bash
git status                                   # rebase or merge? replaying commit? conflicts?
git diff --name-only --diff-filter=U
base=$(git merge-base <feature> <mainline>)
git log --oneline "$base..<mainline>"        # the refactor commits you're landing on
for f in <conflicted files>; do echo "== $f =="; grep -n '^<<<<<<<\|^=======\|^>>>>>>>' "$f"; done
```

For each refactor-looking commit (`move`, `fold`, `refactor … drop`, `thin`),
read just its message: `git show <c> --no-patch --format='%B'`. That's your
migration map. Also scan for a mainline commit that already fixed your bug (e.g.
`fix … evict`) — diff it to avoid a redundant/conflicting fix.

## Step 2 — Resolve conflicts (decision table)

| Conflict shape | Resolution |
|---|---|
| HEAD side empty / your side large (block was **relocated/deleted**) | Take mainline. If *every* change to that file was relocated → `git checkout --ours -- <file>` (whole file). Else resolve that hunk to HEAD. Re-apply your intent in Step 3. |
| Same value, **convention swap** (mainline `getattr(r,"x",0)` vs your `r.y`) | Merge to your value with getattr fallback: `getattr(r, "y", getattr(r, "x", 0))`. |
| Same function, **both really edited** | Keep mainline (`--ours`/HEAD), re-apply your specific edits on top in Step 3. |

> Rebase orientation: `--ours` = the branch you're rebasing **onto** (mainline/HEAD),
> `--theirs` = your replayed commit. Opposite of a normal merge. Verify on one hunk.
> NEVER `--theirs` to "keep my work" — it strands your code where the mainline deleted it.

## Step 3 — Migrate the logic (the real work: non-conflict edits)

Re-apply each feature change at its new home (see the map). Watch for:

- **Signature changed** → update the function AND grep every caller:
  `grep -rn '<fn>(' python/sglang/srt/`. One missed caller = runtime crash.
- **Needed context absent at the new home** → orchestrate from where it exists.
  (e.g. allocator's `compute_dsv4_state_lens_*` has no pool/page_size ref, so drive
  pre-alloc eviction from `mem_cache/common.py::_compute_dsv4_state_lens`, which has
  `batch`, reusing the existing `maybe_evict_dsv4_state`.) Don't force a back-ref
  into a class the mainline deliberately decoupled.
- **getattr/setattr convention** → `req.x = ...` / `getattr(r,"x",default)`; do NOT
  add fields to `Req.__init__` (the refactor removed them on purpose).
- **Common path bit-exact** → branch the feature (`if prefix_len > 0`, `if is_decode`)
  so `prefix_len == 0` / non-feature produces identical offsets/counts to mainline.

The 4 chunked-prefill change sites (compressor metadata, state alloc lens,
pre-alloc eviction, write-offset) are spelled out in `references/dsv4-npu-map.md` §
"Chunked-prefill changes" — apply those, don't re-derive them.

## Step 4 — Verify, then STOP

```bash
python -c "import ast; ast.parse(open('<file>').read())"   # syntax, each touched file
grep -rn '^<<<<<<<\|^>>>>>>>' <touched files>              # zero markers
grep -rn '<changed_fn>(' python/sglang/srt/                # all callers updated
git add <migration files>; git status --short              # conflicts gone, staged
```

**Do NOT run `git rebase --continue` or `git commit`.** This code can't be unit-tested
off-NPU; the human reviews the diff and runs validation first. Leave it staged and
clean, and hand off:

```bash
git diff --cached            # human reviews
git rebase --continue        # human runs after review
# then DSV4 chunked-prefill token-parity (greedy): chunked vs --chunked-prefill-size -1,
# compare output token ids. Identical => migration correct. See the map for details.
```

## Anti-patterns

Keeping your old-location block (dead code → duplicate defs) · `git checkout --theirs`
to "keep my work" · re-adding `Req` fields because getattr felt wrong · changing a
signature and missing a caller · running `rebase --continue` to "finish" (premature —
needs human review + NPU run).
