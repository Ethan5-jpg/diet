# Source Data and precision

Mode: SYNTHETIC_TEST.

Each figure has one CSV under source_data with the exact numerical estimates used for plotting. The tables directory retains all available combinations, including nulls and failures. Native units are mg/dL (Mean/SDw) or percentage points (CV); interactions are per original Diet Z × microbial CLR-Z. Slope estimates are per original Diet Z. No participant rows, participant identifiers, raw input matrices or sample hashes are exported.

SCREENSHOT_PREVIEW uses only recorded displayed values. The 41 unshown primary extension tests, unshown p values and unshown sensitivity coefficients are unavailable, not zero. In this mode, Table S1 contains 43 observed primary rows rather than 84, and the all-extension supplemental forests are omitted. The 17:40 and 19:10 corrections take precedence. Do not use the preview as the final numerical source for submission.

SERVER_FULL_PRECISION reads saved CSV values without recomputing or rounding them before CSV export. Display tables and labels use rounded formatting for readability. Source file hashes are checked before and after export, and the two run identities and aggregate hashes are saved in export_manifest.json. No new model fits or significance tests are performed.

SYNTHETIC_TEST is software validation only; never use it as HPP evidence. Publication use still requires author confirmation of study definitions, particularly rEDIH provenance, and the applicable data-access agreement.

Layout reference: https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/ (accessed 2026-10-10). PDF embeds TrueType fonts; SVG preserves text; PNG is for viewing. Figures use an 183 mm width and at most 170 mm height, following the general Nature figure guide as a design reference, not a claim of journal approval. Each manuscript figure/table can be accompanied by its individual data file in the ZIP, following the source-data organization described at https://www.nature.com/ncomms/submit/how-to-submit . Final numbering and main/supplement allocation depend on the full manuscript.
