# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
