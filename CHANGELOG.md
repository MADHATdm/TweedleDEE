# tweedleDEE Changelog

## [1.0] 2025-09-03

### Added
- New parameter `degree_opening` in `calculate_exposure()` to adjust ROI opening angle.

### Changed
- Renamed `analyze_dwarf()` → `calculate_exposure()` for clarity (reflects purpose: calculating ROI center exposure).
- Updated `calculate_exposure()` to automatically slice the ROI center from the header data file.

### Fixed
- Fixed exposure calculation to average over energy bin edges instead of bin centers (as is default in fermiTools).
- Ensured a newline `\n` is added when appending new dwarfs to `Exposures_updated.tsv` and `IDs_updated.tsv` in `update_exposures()` and `update_IDS()`.