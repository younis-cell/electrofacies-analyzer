# EFA — ElectroFacies Analyzer

EFA is an open, formation-agnostic Streamlit application for exploring well-log data, calculating basic petrophysical curves, and performing unsupervised electrofacies classification from LAS files.

## Run locally

1. Install Python 3.10 or newer.
2. Clone this repository and check out the `dev/efa-general` branch.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Current capabilities

- Load and combine multiple LAS well files.
- Inspect sample counts, depth ranges, and missing-value percentages.
- Calculate VSH (Larionov Tertiary approximation), density porosity (PHID), and water saturation (SW) with configurable Archie parameters.
- Calculate petrophysical curves independently per well.
- Classify electrofacies using K-Means, Gaussian Mixture, or Agglomerative clustering.
- Run clustering independently per well or across the combined dataset.
- Review Silhouette and Davies–Bouldin clustering metrics.
- Plot well-log panels and crossplots.
- Export processed data to CSV and save a portable EFA project archive.

## Scientific cautions

EFA is an analysis aid, not an automatic geological interpreter. Verify LAS curve mnemonics, units, depth references, environmental corrections, and parameter assumptions for every dataset.

- VSH currently uses the Larionov Tertiary approximation. The default clean/shale GR baselines are calculated from each well's P05/P95 values; users may supply explicit baselines.
- PHID uses a single matrix and fluid density. Select values appropriate to the lithology and pore fluid.
- SW uses the clean-formation Archie equation. It may be unsuitable for shaly formations, complex pore systems, or formations requiring dual-water, Indonesia, Simandoux, or other models.
- Unsupervised cluster numbers are arbitrary labels. Validate clusters with core, cuttings, geological context, and independent petrophysical evidence.
- Standardization is performed on the samples used for each clustering run. When wells are clustered independently, label numbers are not necessarily equivalent between wells.

## Project files and security

EFA project files use a ZIP archive containing CSV data and JSON metadata. The application does not deserialize uploaded Python pickle files.

## Development

The `main` branch is retained as the original Zubair-focused version. General-purpose development is taking place on `dev/efa-general`. Contributions and scientific validation are welcome.
