"""Keyboard shortcut management for Lightroom Classic on macOS."""

import subprocess
import sys
from pathlib import Path

LIGHTROOM_BUNDLE_ID = "com.adobe.LightroomClassicCC7"

# NSUserKeyEquivalents modifier key symbols:
# @ = Command, $ = Shift, ~ = Option, ^ = Control
SHORTCUTS = {
    "Auto Crop": "~$r",  # Option+Shift+R
    "Settings": "^g",  # Control+G
}


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


def install_shortcuts() -> bool:
    """Install macOS keyboard shortcuts for Lightroom Classic."""
    if sys.platform != "darwin":
        print("Shortcuts installation is only supported on macOS.", file=sys.stderr)
        return False

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
    if sys.platform != "darwin":
        print("Shortcuts removal is only supported on macOS.", file=sys.stderr)
        return False

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
    plist_path = Path.home() / f"Library/Preferences/{LIGHTROOM_BUNDLE_ID}.plist"
    for menu_item in SHORTCUTS.keys():
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
