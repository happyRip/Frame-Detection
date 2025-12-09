# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- JSON-based filter configuration via `--filter-config` CLI argument
- FilterConfig dataclasses for structured parameter management
- Configurable parameters for all edge detection filters:
  - Canny: low/high thresholds
  - Sobel/Scharr/Laplacian: blur size
  - DoG: sigma1/sigma2
  - LoG: sigma
- Configurable parameters for all separation methods:
  - All methods: tolerance
  - CLAHE: clip limit, tile size
  - Adaptive: block size
  - Gradient: gradient weight
- New "Filters" tab in Lightroom plugin dialog with:
  - Edge filter selection and parameter tuning
  - Separation method selection and parameter tuning
  - Preview button placeholder for future live preview feature

## [0.0.6] - 2025-12-09

### Changed
- CLI now requires explicit `detect` subcommand for frame detection

## [0.0.4] - 2025-12-09

### Fixed
- CLI no longer requires OpenCV for `install`/`uninstall` commands (lazy imports)

### Added
- Auto-discovery of `negative-auto-crop` command from Homebrew locations
- Configurable command path in plugin Debug settings

## [0.0.3] - 2025-12-09

### Changed
- Installation method changed to Homebrew distribution
- CLI tool and Lightroom plugin now installed separately
- Keyboard shortcuts managed via `negative-auto-crop install/uninstall shortcuts`

### Removed
- Bundled Python environment in plugin (now uses system command)

### Migration
If upgrading from a previous version, uninstall first:

```bash
negative-auto-crop uninstall
```

See [homebrew-negative-auto-crop](https://github.com/USER/homebrew-negative-auto-crop) for installation instructions.
