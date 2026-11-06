# Integrity exceptions

## Development exposure of a frozen family, Nov 7 IST

During environment and file-schema inspection, the coding agent printed the
header and first row of the mixed `vedant_review_nov7.csv` queue. That row was
`rev-b01-001`, canonical source `xstest:26`, family `fam_983e49e2566f`, split test.
The prompt and its language variants therefore appeared in development context.
No model response or test score was produced. This is an exposure, and the
existing frozen manifest cannot be described as wholly unseen by the agent.

The exposure is recorded in `data/splits/development_exposures.json`. The old
manifest remains byte-for-byte unchanged. The locked-test guard refuses a
manifest that still assigns an exposed family to test. Before the final pass,
the Research Lead must review a new versioned holdout and preregister any
necessary exclusion or replacement. Do not silently rewrite the frozen file,
move the exposed prompt into development, or count its removal as a safety gain.

The committed calibration/dev review sheet was rebuilt with canonical
source-member validation and is identical to the prior sheet. The source
membership bug reported in council round 2 was reproduced with synthetic text;
that repro did not establish leakage into the committed development sheet.

Council also disclosed that an initial invariant-test command parsed mixed-split
staging CSVs. Those checks inspect membership rather than model behavior, and
the requested family-split checker itself parses those CSVs. This distinction
does not excuse printing a test prompt in the earlier header inspection.
