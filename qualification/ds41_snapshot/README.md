# Prepared FI source snapshots

This branch preserves the push100 FI source and tests used by the DS4.1 TP4
resident assembly. Its all-reduce uses push reduce-scatter followed by push
all-gather from 100 KiB through the caller's channel capacity, including the
original scratch allocation. It does not adopt the 64 KiB dispatch threshold
or storage changes from subsequent FI revisions.

The assembly also retains separately prepared FI implementations. Each patch
below is independent and applies to `24407e1cbe41a325baa9a23e98ace2fa0c5f472e`,
not on top of this branch or another patch.

| Patch | Prepared implementation |
|---|---|
| `variants/collective-capability.patch` | Unchanged base kernels plus the retained-session preparation helper; cooperative CSF decoding without metadata preload and the base gather implementation. |
| `variants/csf-phase1.patch` | FC1 bounds/header/base preload with cooperative FC2 decoding. |
| `variants/csf-phase2-warp.patch` | FC1 metadata preload with consumer-warp FC2 decoding. |
| `variants/dcp-push.patch` | Push all-gather kernel with compile identity 14; base host geometry remains unchanged. |
| `variants/pull100.patch` | Retained pull all-reduce comparator with the 100 KiB threshold. |
| `variants/pull256.patch` | Retained pull all-reduce comparator with the 256 KiB threshold. |

For example, materialize the FC1-preload/cooperative-FC2 implementation from
the repository root:

```sh
git worktree add --detach ../fi-csf-phase1 24407e1cbe41a325baa9a23e98ace2fa0c5f472e
git -C ../fi-csf-phase1 apply --unidiff-zero "$(pwd)/qualification/ds41_snapshot/variants/csf-phase1.patch"
```

Use a separate worktree and the corresponding patch for each other variant.
The matching Flare assembly chooses these prepared implementations by graph
role; this snapshot introduces no global shape rule or plan override.

The source implementations have existing component and resident qualification.
This publication preserves their bytes and verifies patch reconstruction; it
does not claim a fresh GPU run or that importing this branch alone recreates
the complete model assembly. Model recipes, prepared objects and serving
configuration are separate from these FI sources.
