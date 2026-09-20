# M1 offline evidence

See the [experiment record](../../head-stage-calibration-20260920.md) for the
fixed protocol, commands, results and limits. All compute ran on the evaluation
server with its existing environment; no local model/test execution or install.

| Phase | Frozen source | Result |
|---|---|---|
| Input capture / original probe tests | `ea5ac5f` | Three real inputs; 54 tests passed |
| Full intervention sweep / original analysis | `7c79c96` | 6,480 interventions; exit 0 |
| Dense AV/VV statistics / expanded analysis | `f489335` | 21,600 step/head statistics; 9 tests passed; exit 0 |
| Final-step causal control | `42cd75b` | 72/72 unchanged final actions with changed video; exit 0 |
| Matched-mask-budget combined profiles | `1adeeff` | 21 records, repeated controls; 13 tests passed; exit 0 |
| Complete matrix/classification audit | `90b3b77` | All identities/scope/labels and raw action metrics verified; exit 0 |
| Stability plot export | `b6847cd` | PNG/PDF rendered and visually inspected; exit 0 |

`head-stage.csv` contains every unit's relative type, continuous calibration
and check sensitivities, and aligned AV/VV statistics. `classification-summary`
omits the duplicated 2,160-entry JSON array; it records the full source JSON
hash. `combined-report.json` retains all seven profiles and every raw predicted
action, including all three random controls. The prefix audit additionally
reports the first ten denormalized actions actually executed by the protocol.

The large intervention JSONL and reference/input NPZ files remain in
`outputs/head-stage-calibration-20260920/` on the evaluation server, addressed
by their manifests and SHA256 hashes. These selected local evidence copies
were verified byte-for-byte by SHA256 against the server:

| Artifact | SHA256 |
|---|---|
| Full intervention matrix (server) | `5d4a854d5024b683ad94ad0e8ea82f42aded13a83a131bb923e067b02f4cdb10` |
| Full manifest | `003ac9fc8d59f620dc1832f88693552523746450cb7f87057a6e1eec12d1a0fe` |
| Full audit | `7f233aaf90be576aaa9ccaa9bf96f9b2bf4d2d22d4ecde84decc22ec09b99dae` |
| Type/metric CSV | `333aa8e5225f9d48114311ae52f018a68ddd591b05a7d887dc9488bc555e1bc2` |
| Combined profiles/actions | `f9eb4a841efb1a929e8568b45bb7be5f64c6c7e32eada3079e891d8b4d39b11e` |

The three official initial observations are exposed calibration/check inputs.
No file in this evidence folder establishes benchmark SR, SR tolerance or an
optimized sparse operator speedup. Equal mask cardinality is not equal measured
kernel cost or proof of skipped FLOPs. Current fast SR policies remain distinct.
