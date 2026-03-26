# tweedleDEE Changelog

## [1.1] 2026-03-26

### Added
- Command-line interface with `argparse` for flexible analysis configuration
- Type hints throughout all functions for better code clarity
- Comprehensive docstrings in NumPy format for all functions
- Structured logging system with separate log file (`testing.log`)
- `FermiBackgroundResult` dataclass for organizing Fermi background analysis results
- `PMFConfig` dataclass for PMF configuration with YAML loading support (`from_yaml()` classmethod)
- `CoordinateSet` dataclass for organizing target, source, sample, and extended coordinates
- `PMFResults` dataclass for PMF data arrays (pmf_data, pmf_list, means, n_obs, photon_counts, max_counts)
- `LikelihoodResults` dataclass for independent and covariance likelihood values with delta
- `EnergyBinResult` dataclass for per-bin PMF histogram and photon count data
- `L_indep()` method for independent log-likelihood calculation using PMF probabilities
- `L_cov()` method for multivariate Gaussian log-likelihood with covariance matrix
- `_setup_coordinates()` private method for coordinate preparation workflow
- `_process_energy_bin()` private method for single energy bin PMF processing
- `_generate_pmf_all_bins()` private method for multi-bin PMF generation
- `_build_pmf_array()` private method to construct 2D PMF array with padding
- `_calculate_nobs()` private method for observed counts calculation in target region
- `PlotSkyRegion` class with `plot_sky_coords_comprehensive()` for sky region visualization
- Extended source tracking and mask flags in `get_source_coords()`
- Logging system with `logging` module (INFO level) in configSetup.py
- FileNotFoundError handling in `loadYAML()` for missing config files
- Command-line interface in configSetup.py with `argparse` when run as standalone script
- Support for targets.yaml (non-default sky locations) in addition to defaults files
- ltcube file reuse detection in `setup_config_yaml()` to avoid redundant computation
- Timeout parameter (30s) for HTTP requests in `get_file()`
- Return type (bool) for `get_file()` indicating download success/failure
- Absolute path resolution in events.txt using `Path.resolve()`
- Dual config save to both input and output directories
- Parent directory auto-creation with `Path.mkdir(parents=True, exist_ok=True)`
- `count_free_parameters()` function to track model complexity in ROI
- `roi_fit()` function for complete ROI fitting workflow with residual map generation
- `calculate_BIC()` and `calculate_AIC()` functions for model comparison metrics
- `fermi_background()` function for Fermi-LAT background likelihood calculation
- Integration with `BgdModelAnalysis` for PMF generation and likelihood analysis
- Configuration file support with `--fermi_config_file` and `--td_config_file` options
- Complete analysis pipeline in `main()` comparing PMF (independent/covariance) vs Fermi background models
- ROI postfit caching to avoid redundant fitting
- Circular aperture extraction for counts and model maps using astropy units
- Version constraints for fermipy (~=1.4) to ensure compatibility

### Changed
- Renamed "dwarf" terminology to "sky_location" throughout for generality
- Renamed `createPMF` class to `BgdModelAnalysis` with updated API
- Renamed `get_dwarfs()` to `get_sky_locations()` for consistency
- Refactored `create_PMF()` to return `PMFResults` and `LikelihoodResults` dataclasses
- Updated `get_target_coords()` to use ICRS frame instead of Galactic frame (changed `l,b` to `ra,dec`)
- Modified `get_source_coords()` to return extended source flags and expanded search radius to `target_size + source_size + 0.5`
- Updated `get_event_coords()` to convert fk5 coordinates to ICRS frame
- Enhanced `create_PMF_values()` to return raw photon count array (`hist_event`) for covariance analysis
- Reorganized `create_PMF()` workflow into modular private helper methods for maintainability
- Disabled PMF file output in `save_PMF()` (file saving code commented out)
- Restructured `generate_PMF()` as public API wrapper around internal `create_PMF()` workflow
- Updated initialization to use `PMFConfig` dataclass with validation in `__post_init__()`
- Changed `dwarf_files_dir` parameter to `sky_location_files_dir` for consistency
- Replaced all `os.chdir()` and `os.getcwd()` calls with `pathlib.Path` operations throughout configSetup.py
- Renamed `dwarf` parameter to `sky_location` in `setup_config_yaml()`
- Extended `setup_config_yaml()` signature with `input_dir`, `output_dir`, `config_file`, and `defaults` parameters
- Modified `get_file()` to return bool instead of None and use `response.raise_for_status()`
- Enhanced `configure_input_files()` to accept directory and config file parameters
- Replaced `glob.glob()` with `Path.glob()` for file pattern matching
- Changed all `print()` statements to `logging.info()`, `logging.warning()`, or `logging.error()`
- Updated `get_catalogs()` to use raw.githubusercontent.com URL for direct file downloads
- Modified `saveYAML()` to use `sort_keys=False` preserving key ordering
- Updated `main()` to perform full analysis pipeline with BIC/AIC comparison
- Changed `runGTA()` to return GTAnalysis object for downstream use
- Modified `calculate_exposure()` to work with generic sky locations
- Removed parallel processing code (ProcessPoolExecutor) for simplified workflow
- Reorganized code structure with analysis functions grouped logically
- Updated `update_exposures()` and `update_IDs()` to use `get_sky_locations()` instead of `get_dwarfs()`
- Updated `configSetup.py` to accept and propagate `targets_file` parameter through `setup_config_yaml()` and `configure_input_files()`
- Replaced local variable names in configSetup.py to use `targets_file` parameter instead of hardcoded `'targets.yaml'`

### Removed
- Removed parallelization; users now directed to HPC for batch processing multiple targets

- **Environment (environment.yml):**
  - Python version: `>=3.9` → `~=3.11.11` (pinned to Python 3.11.11 with compatible releases)
  - fermitools: `>=2.2.0` → `~=2.4.0` (upgraded to 2.4.0)
  - astropy: `=5.3.4` → `~=7.0.1` (major upgrade to 7.0.1 for improved compatibility)
  - Package pinning strategy: changed from exact (`=`) and minimum (`>=`) to compatible release (`~=`) for numpy, pyyaml, and requests

## [1.0] 2025-09-03

### Added
- New parameter `degree_opening` in `calculate_exposure()` to adjust ROI opening angle.

### Changed
- Renamed `analyze_dwarf()` → `calculate_exposure()` for clarity (reflects purpose: calculating ROI center exposure).
- Updated `calculate_exposure()` to automatically slice the ROI center from the header data file.

### Fixed
- Fixed exposure calculation to average over energy bin edges instead of bin centers (as is default in fermiTools).
- Ensured a newline `\n` is added when appending new dwarfs to `Exposures_updated.tsv` and `IDs_updated.tsv` in `update_exposures()` and `update_IDS()`.