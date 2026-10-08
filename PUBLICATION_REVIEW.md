# Publication review

| Material | Finding | Recommended action |
| --- | --- | --- |
| HW1/HW2 reports, PDF and DOCX | Names/student numbers | Approve attribution; redact identifiers in separate publication copies if authorized |
| HW1 assignment/guide | Instructor material and complete solutions | Confirm course policy and copyright |
| 2/code/Covid.csv | Patient health/demographic/comorbidity features; source/license absent | Review privacy, de-identification and redistribution |
| 3/processed/model_tables/ | Host/customer IDs, coordinates, ZIPs, employee features | Review source rights/re-identification |
| 3/cache/*.pkl | Sample may retain richer raw fields | Trusted isolated review if needed; not deserialized here |
| Saved game weights | Small serialized models | Trusted provenance before loading |
| HW3 notebook | Absolute local paths; no saved outputs | Review public local-path exposure |
| Bytecode/sampling cache | Disposable generated files | Ignore covers future additions; tracked content stays tracked |

Student numbers and credential values were not copied into generated artifacts. Attribution names come from reports; collaborator permission remains to be confirmed.

## Scan scope

Heuristic path-only scanning covered readable Python, Markdown, notebook JSON, CSV, JSON and text, plus extracted PDF/DOCX text. No high-confidence credential assignments or recognizable key-format matches were found. No environment/credential files or virtual environments are present in the coursework inventory. This is not proof of absence: image-only text, arbitrary binaries, pickle contents and Git history were not comprehensively scanned. No pickle was loaded.

## Size findings

Largest file: 3/processed/model_tables/nyc311_resolution_regression_sample.csv, 7,936,160 bytes (7.57 MiB). Sampling pickle: 5,416,645 bytes. No existing coursework file exceeds 50 MiB or 100 MiB; no large neural checkpoint was found. The report's roughly 2.8 GB NYC raw input is absent.

## Before public release

Confirm collaborator consent, course policy, report redaction, source ownership and dataset rights. Consider a private repository while unresolved. Ignore rules do not remove tracked content or history. Nothing was deleted, untracked, committed, pushed or published. Optional exclusions are in .gitignore.portfolio-suggestions; review the license notice before choosing a scoped code license.
