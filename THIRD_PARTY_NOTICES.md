# Third-party notices

This repository contains code and derived data from several sources.  The MIT
license in `LICENSE` applies to the original code in this repository only.  It
does not relicense third-party data or software.

## MPTP2078 / Mizar

`data/mptp`, `data/cnf`, and `data/atp/statements` are derived from MPTP2078
and the Mizar Mathematical Library.

Upstream project: https://github.com/JUrban/MPTP2078

Upstream licensing allows GPL-3.0-or-later or CC-BY-SA-3.0-or-later, subject to
`COPYING.interpretation`.  The upstream license files are included verbatim in
`third_party/mptp2078/`:

- `COPYING`
- `COPYING.GPL`
- `COPYING.CC-BY-SA`
- `COPYING.interpretation`

If you redistribute the derived data, keep these files, attribute MPTP2078 and
Mizar, and preserve the upstream share-alike/notice requirements.  If you only
redistribute code, remove `data/` and download MPTP2078 from upstream.

## ATPBoost

The ATP feedback protocol and training-data construction follow ATPBoost.
ATPBoost is distributed under the MIT license; the upstream license text is in
`third_party/atpboost/LICENSE`.

## E-prover

The proof-runner scripts call an external E-prover executable.  E-prover is not
bundled in this repository.  It is distributed under GPL-2.0-or-later /
LGPL-2.1-or-later options; the upstream license text is included in
`third_party/eprover/COPYING` for reference.  Install E-prover 3.2.5 separately
and comply with its license.

## Other software

PyTorch, PyTorch Geometric, PyTorch Scatter, lark, NumPy, pandas, NetworkX,
Matplotlib, and tqdm are installed as dependencies and are not vendored here.
Their own licenses apply.
