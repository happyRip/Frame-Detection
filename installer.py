#!/usr/bin/env python3
"""Plugin and shortcut installation for macOS."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_NAME = "NegativeAutoCrop.lrplugin"
LIGHTROOM_MODULES_DIR = Path.home() / "Library/Application Support/Adobe/Lightroom Classic/Modules"
PLUGIN_DIR = LIGHTROOM_MODULES_DIR / PLUGIN_NAME

LIGHTROOM_BUNDLE_ID = "com.adobe.LightroomClassicCC7"

# NSUserKeyEquivalents modifier key symbols:
# @ = Command, $ = Shift, ~ = Option, ^ = Control
SHORTCUTS = {
    "Auto Crop": "~$r",  # Option+Shift+R
    "Settings": "^g",    # Control+G
}


def get_script_dir() -> Path:
    """Get the directory where this script lives."""
    return Path(__file__).parent.resolve()


def get_plugin_source_dir() -> Path:
    """Get the source directory for plugin Lua files."""
    script_dir = get_script_dir()

    # Plugin dir should be sibling to this script
    plugin_dir = script_dir / "plugin"
    if plugin_dir.exists():
        return plugin_dir

    raise FileNotFoundError(f"Plugin source directory not found at {plugin_dir}")


def get_frame_detection_dir() -> Path:
    """Get the frame_detection package directory."""
    script_dir = get_script_dir()

    package_dir = script_dir / "frame_detection"
    if package_dir.exists():
        return package_dir

    raise FileNotFoundError(f"frame_detection package not found at {package_dir}")


def install_plugin() -> bool:
    """Install the Lightroom plugin to the Modules directory."""
    try:
        plugin_source = get_plugin_source_dir()
        package_dir = get_frame_detection_dir()
        script_dir = get_script_dir()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    print(f"Installing plugin to {PLUGIN_DIR}...")

    # Create Modules directory if it doesn't exist
    LIGHTROOM_MODULES_DIR.mkdir(parents=True, exist_ok=True)

    # Remove existing plugin if present
    if PLUGIN_DIR.exists():
        shutil.rmtree(PLUGIN_DIR)
        print("  Removed existing plugin")

    PLUGIN_DIR.mkdir()

    # Copy Lua files
    for lua_file in plugin_source.glob("*.lua"):
        shutil.copy2(lua_file, PLUGIN_DIR)
        print(f"  Copied {lua_file.name}")

    # Copy Python package (frame_detection/)
    frame_detection_dst = PLUGIN_DIR / "frame_detection"
    shutil.copytree(package_dir, frame_detection_dst)
    print("  Copied frame_detection/")

    # Copy requirements.txt if it exists
    requirements_src = script_dir / "requirements.txt"
    if requirements_src.exists():
        shutil.copy2(requirements_src, PLUGIN_DIR)
        print("  Copied requirements.txt")

    # Create virtual environment
    print("Creating Python virtual environment...")
    venv_path = PLUGIN_DIR / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    # Install dependencies
    pip_path = venv_path / "bin" / "pip"
    requirements_dst = PLUGIN_DIR / "requirements.txt"
    if requirements_dst.exists():
        print("Installing Python dependencies...")
        subprocess.run(
            [str(pip_path), "install", "-q", "-r", str(requirements_dst)],
            check=True,
        )

    print(f"\nPlugin installed successfully!")
    print(f"Location: {PLUGIN_DIR}")
    print("\nRestart Lightroom Classic to load the plugin.")
    return True


def uninstall_plugin() -> bool:
    """Remove the Lightroom plugin."""
    if PLUGIN_DIR.exists():
        shutil.rmtree(PLUGIN_DIR)
        print(f"Plugin uninstalled from {PLUGIN_DIR}")
        print("\nRestart Lightroom Classic to complete removal.")
        return True
    else:
        print("Plugin not found, nothing to uninstall.")
        return False


def install_shortcuts() -> bool:
    """Install macOS keyboard shortcuts for Lightroom Classic."""
    print("Installing keyboard shortcuts for Lightroom Classic...")

    for menu_item, shortcut in SHORTCUTS.items():
        subprocess.run(
            [
                "defaults",
                "write",
                LIGHTROOM_BUNDLE_ID,
                "NSUserKeyEquivalents",
                "-dict-add",
                menu_item,
                shortcut,
            ],
            check=True,
        )
        print(f"  {menu_item}: {_shortcut_display(shortcut)}")

    print("\nShortcuts installed!")
    print("Restart Lightroom Classic to apply.")
    return True


def uninstall_shortcuts() -> bool:
    """Remove macOS keyboard shortcuts for Lightroom Classic."""
    print("Removing keyboard shortcuts from Lightroom Classic...")

    # Read current shortcuts
    result = subprocess.run(
        ["defaults", "read", LIGHTROOM_BUNDLE_ID, "NSUserKeyEquivalents"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("No shortcuts found, nothing to remove.")
        return False

    # Remove each shortcut we installed
    for menu_item in SHORTCUTS.keys():
        # Use PlistBuddy to delete specific keys
        plist_path = Path.home() / f"Library/Preferences/{LIGHTROOM_BUNDLE_ID}.plist"
        subprocess.run(
            [
                "/usr/libexec/PlistBuddy",
                "-c",
                f"Delete :NSUserKeyEquivalents:'{menu_item}'",
                str(plist_path),
            ],
            capture_output=True,
        )
        print(f"  Removed: {menu_item}")

    print("\nShortcuts removed!")
    print("Restart Lightroom Classic to apply.")
    return True


def _shortcut_display(shortcut: str) -> str:
    """Convert shortcut code to human-readable format."""
    parts = []
    if "^" in shortcut:
        parts.append("Control")
    if "~" in shortcut:
        parts.append("Option")
    if "$" in shortcut:
        parts.append("Shift")
    if "@" in shortcut:
        parts.append("Cmd")

    # Get the actual key (last character)
    key = shortcut[-1].upper()
    if key == ",":
        key = "Comma"
    parts.append(key)

    return "+".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Install/uninstall NegativeAutoCrop Lightroom plugin and shortcuts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # install command
    install_parser = subparsers.add_parser("install", help="Install components")
    install_subparsers = install_parser.add_subparsers(dest="target")
    install_subparsers.add_parser("plugin", help="Install the Lightroom plugin")
    install_subparsers.add_parser("shortcuts", help="Install keyboard shortcuts")

    # uninstall command
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall components")
    uninstall_subparsers = uninstall_parser.add_subparsers(dest="target")
    uninstall_subparsers.add_parser("plugin", help="Remove the Lightroom plugin")
    uninstall_subparsers.add_parser("shortcuts", help="Remove keyboard shortcuts")

    args = parser.parse_args()

    # If no target specified, run both plugin and shortcuts
    if args.target is None:
        if args.command == "install":
            success = install_plugin() and install_shortcuts()
        else:
            success = uninstall_shortcuts() and uninstall_plugin()
    else:
        commands = {
            ("install", "plugin"): install_plugin,
            ("install", "shortcuts"): install_shortcuts,
            ("uninstall", "plugin"): uninstall_plugin,
            ("uninstall", "shortcuts"): uninstall_shortcuts,
        }
        success = commands[(args.command, args.target)]()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
