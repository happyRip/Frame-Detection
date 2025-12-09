# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- New `--sprocket-type` CLI option to control sprocket hole detection separately from film type:
  - `none` - skip sprocket detection (medium format, pre-cropped images)
  - `bright` - expect bright sprocket holes
  - `dark` - expect dark sprocket holes
  - `auto` - auto-detect (default)
- Plugin: New "Sprocket type" setting in Film tab

## [0.0.8] - 2025-12-09

### Changed
- Reorganized plugin settings dialog: new "Film" tab as first tab with film type and auto crop buttons, "Crop" tab now contains only crop-related settings
- Moved "Restore Defaults" button to bottom left of main dialog window
- Navigation buttons disabled when single photo selected

## [0.0.7] - 2025-12-09

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
  - Film Base and Edge Detection subtabs
  - Edge filter selection and parameter tuning
  - Separation method selection and parameter tuning
  - Reset buttons (↺) next to parameter labels to restore defaults
  - Single dynamic preview button that switches between "Preview Film Base" and "Preview Edges" based on selected subtab
- Preview backup flow with `.bak` rotation for preview temp directories
- `--preview-mode` CLI argument for minimal preview generation:
  - `separation`: generates only film base mask visualization
  - `edges`: generates only edge detection visualization

### Fixed
- Renamed JSON.lua to JsonEncoder.lua to avoid Lightroom reserved name conflict

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
