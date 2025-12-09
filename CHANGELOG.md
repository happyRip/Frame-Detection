# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
