# Review bundle: sc26-pyhpc-v2 (c1fdc91) against the head of PR #296 (2d0d06d)

Temporary branch, not for merging. The commit below this one carries the tree of `sc26-pyhpc-v2` at `c1fdc91`, so GitHub's
compare view of `sc26-pyhpc` against that commit shows every change since the PR head. GitHub pairs the kernel sources as
renames but shows `cosmic_core/fields.py` and `cosmic_core/zeldovich.py` as new files; the diffs here pair them.

- `stat.txt`: `git diff -M20% --stat`, renames paired at 20 % similarity.
- `code.diff`: everything except notebooks, same pairing: `fields.py` against `testfield.py`, `zeldovich.py` against
  `dataset.py`, `ecc_histogram.cu` against `cosmic_ecc.cu`, `ecc_histogram_rows.cu` against `cosmic_ecc_variants.cu`,
  `cub_histogram.cu` against `cosmic_cub_histogram.cu`.
- `notebooks.diff`: the notebooks rendered as cell text (sources only, one block per cell). 90, 95 and the syllabus are
  compared with themselves; 91 to 94 are compared with the SOLUTION notebooks of the PR head, which the single set came from.
  Cell numbers in the headers are those of the rendering, counted from 0.
- `BASE.txt`: the two commits and the time of the build.
