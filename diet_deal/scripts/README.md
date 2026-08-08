# Diet processing scripts

Scripts are separated into shared preparation and score-specific pipelines.

- `shared/`: common daily summaries and the reviewed food dictionary.
- `amed/`: AMED mapping, component-intake aggregation, and scoring.
- `ahei/`: AHEI mapping, component-intake aggregation, and scoring.
- `hpdi/`: hPDI mapping, component-intake aggregation, and scoring.
- `rdii/`: rDII nutrient preparation and scoring.
- `redih/`: rEDIH mapping, component-intake aggregation, and scoring.

Each script resolves source data and output paths from the `diet_deal` project
directory, so it can be run from any working directory.
