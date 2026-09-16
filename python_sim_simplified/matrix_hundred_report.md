# Matrix calibration research 041–060

Twenty distinct model/algorithm research closures. Each round has proposal.json, results.json, conclusion.json and actual Python source snapshots. The initial sequence was run one round at a time; the next proposal followed inspection of the preceding result. Parameter sweeps, optimization iterations, regression runs and new seeds are not counted as research rounds.

## Geometry correction and evidence provenance

The first sequence incorrectly used a 0.035 × 0.075 m array and a square distribution of horizontal direction cosines that could leave the physical unit disk. Root review identified this error. All original artifacts were preserved under results/research-hundred/matrix_initial_geometry. The same 20 mechanisms were then rerun with the exact project Mate70 TL/TR/BL/BR phase centers, 0.0759 × 0.1609 m, uniform azimuth and uniform sin(elevation) from 15 degrees to asin(0.97). The original erroneous geometry results are historical evidence, not final performance.

Correction changes the angular-design condition from 862.5 to 35.7 and substantially reduces the earlier extreme two-antenna noise amplification. Those earlier strong conclusions are withdrawn. The rank/nullspace nonidentifiability findings remain. Round 058 was originally motivated by the ill-conditioned initial model; after correction it has only marginal development benefit and a holdout reversal, so it is not recommended as a uniform improvement.

Development seed is 41060. Original-geometry verification seeds 51061/51062 are archived and are not reused as fresh verification. Final corrected algorithms were hash-frozen before new seeds 61061/61062: 40 model evaluations, zero algorithm changes afterward. These are new synthetic realizations within the same model family, not an independent RF or hardware distribution.

## Per-round index

| Round | Actual model/algorithm change | Corrected development result | Decision |
|---|---|---|---|
| 041 | [Full projective matrix](matrix_hundred_041.py) | 17.809% diagonal -> 5.26e-16 full; authentic input assumed. | retain conditional |
| 042 | [Source-only rank degeneracy](matrix_hundred_042.py) | 8.52e-16 false-fit residual; rank 1; nullity 4 versus authentic 1. | reject residual-only acceptance |
| 043 | [Projected full matrix](matrix_hundred_043.py) | Nullity 5; identity prior heldout error 22.109%. | reject blind full-matrix claim |
| 044 | [Reciprocal matrix structure](matrix_hundred_044.py) | Symmetric matrix still has nullity 2; invisible v v-transpose component. | reject reciprocity as sufficient |
| 045 | [Independent multiport pilots](matrix_hundred_045.py) | Full 0.386%, diagonal 20.607%, common-tone 82.961%; pre-network injection assumed. | retain conditional hardware route |
| 046 | [Sign-reversed injection](matrix_hundred_046.py) | Constant ambient: 404.110% direct -> 0.255% differential. | retain under stationary nuisance |
| 047 | [Affine ambient nuisance](matrix_hundred_047.py) | 8.202% differential -> 0.511% affine joint fit. | retain under affine nuisance |
| 048 | [Curved ambient nuisance](matrix_hundred_048.py) | 1.289% affine -> 0.511% quadratic joint fit; pilot-coherent interference inseparable. | retain under quadratic nuisance |
| 049 | [Frequency-selective matrix](matrix_hundred_049.py) | 180ns tap: 15.424% flat -> 1.881% individually measured subbands. | retain frequency model |
| 050 | [Frequency-affine smoothing](matrix_hundred_050.py) | Noisy heldout tones: 10.907% interpolation -> 9.149% affine. | retain provisional smoothing |
| 051 | [Shared unknown delay tap](matrix_hundred_051.py) | 9.149% affine -> 8.053% delayed fit; estimated 210.4ns, simulated 180ns. | retain conditional structural model |
| 052 | [Causal recurring matrix update](matrix_hundred_052.py) | 60s mismatch 50.077% static -> 1.443% online; online maximum 1.984%. | retain with recurring independent pilots |
| 053 | [Reference-free full fit](matrix_hundred_053.py) | Weak channel ratio error 15.609% -> 0.578% reference-free. | retain with authentic input |
| 054 | [Unknown reference delay](matrix_hundred_054.py) | Equivalent decompositions differ by 2.96e-16; 120ns = 35.975m common range, not horizontal error. | reject absolute-delay claim |
| 055 | [Independent coherent reference monitor](matrix_hundred_055.py) | 46.239% unmonitored -> 0.624% monitored; delay error 0.027ns in synthetic model. | retain with characterized monitor path |
| 056 | [Injection point / OTA fixture](matrix_hundred_056.py) | Post-network internal pilot misses coupling: 20.723% -> 0.123% with explicit OTA data. | require external end-to-end measurement |
| 057 | [Angular matrix basis](matrix_hundred_057.py) | Flat 15.237% -> angular 0.108%; design condition 35.675 after geometry correction. | retain for stated fixture model |
| 058 | [Angular coefficient prior](matrix_hundred_058.py) | 2.168% unregularized -> 2.076% prior; holdout 61062 worsens 2.884% -> 2.965%. | do not recommend as uniform improvement |
| 059 | [Measured covariance constraints](matrix_hundred_059.py) | 2/3/4 antennas noise ratios 1.000/0.178/0.166; two constraints exhaust two-antenna freedom. | retain when spatial freedom remains |
| 060 | [Nested source-information comparison](matrix_hundred_060.py) | Absent/authentic residuals nearly identical; absent aggregate BIC support -151/-172/-187; one-PRN false positives remain. | retain evidence comparison, not navigation claim |

## Receiver information contract and limits

All estimator functions receive measured complex outputs plus nominal or explicitly commanded inputs. Simulation truth h, test_y, and genuine flags are used to synthesize data or evaluate errors, never substituted for observed calibration inputs. Authentic sky calibration in 041/053 explicitly assumes genuine signals; it does not establish authenticity. Pilot-dependent routes explicitly add simulated, independently commanded measurements; they are not improvements available from the same ambient GNSS data alone.

Independent multiport injection before the modeled coupling network, coherent monitor path characterization, known-amplitude OTA fixtures, angular priors and recurring pilots are unverified external hardware assumptions. We have no bench data. Round 056 explicitly shows that an internal injection after antenna coupling cannot identify that coupling. Frequency models use +/-1 MHz tone observations and are not an RF tracking validation. Round 060 assumes an ideal measured source-only training block and known noise level; nested evidence is not an integrity bound.

No PVT was run in this package. <=20 m, >=30 m and >=50 m position counts, sustained availability and few-meter positioning are not measured and must not be inferred from response errors. The common-delay range equivalent in 054 can enter receiver clock and is not horizontal position error. In the absent-authentic control there is no authentic range information, so this package supplies no trustworthy continuing navigation.

## Frozen validation and failures

All 20 corrected snapshots reproduced their result JSON exactly and their source SHA256 hashes matched. Five focused tests passed. Frozen new-seed results preserve the full-matrix and reference-free improvements, projected nullity 5, reciprocal nullity 2, and no noise optimization freedom for two antennas. Round 058 fails to improve seed 61062 (2.884% -> 2.965%). Round 060 has occasional false individual-PRN support despite negative absent aggregate evidence; do not treat an individual model-fit gain as certified authenticity.

Evidence: results/research-hundred/matrix_freeze.json, matrix_holdout.json, matrix_reproduction.json, matrix_tests.log. Corrected per-round executions are logged in matrix_corrected_runs.log. The complete original evidence is under matrix_initial_geometry.

Reproduce from python_sim_simplified with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1, using D:/Python/Python311/python.exe -m pytest tests/test_matrix_hundred.py -q. Individual development models: python matrix_hundred_core.py 41 (through 60). Snapshot reproduction and holdout evaluation: python matrix_hundred_validate.py; rerunning writes a new freeze timestamp and the same deterministic results, so preserve existing evidence first when needed.
