#!/bin/bash
set -e

# Release script: updates versions, commits, tags, and pushes

show_help() {
    echo "Usage: $0 [OPTIONS] [VERSION]"
    echo ""
    echo "Options:"
    echo "  -p, --patch    Increment patch version (X.Y.Z -> X.Y.Z+1) [default]"
    echo "  -m, --minor    Increment minor version (X.Y.Z -> X.Y+1.0)"
    echo "  -M, --major    Increment major version (X.Y.Z -> X+1.0.0)"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "If VERSION is provided, it overrides any increment flag."
    echo ""
    echo "Examples:"
    echo "  $0 --patch           # Increment patch: 0.0.3 -> 0.0.4"
    echo "  $0 -m                # Increment minor: 0.0.3 -> 0.1.0"
    echo "  $0 --major           # Increment major: 0.0.3 -> 1.0.0"
    echo "  $0 0.0.5             # Use explicit version 0.0.5"
    echo "  $0 --minor 0.0.5     # Use explicit version 0.0.5 (flag ignored)"
}

# Get current version from pyproject.toml
get_current_version() {
    grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'
}

# Increment version based on type
increment_version() {
    local version="$1"
    local type="$2"

    local major minor patch
    IFS='.' read -r major minor patch <<< "$version"

    case "$type" in
        major)
            echo "$((major + 1)).0.0"
            ;;
        minor)
            echo "${major}.$((minor + 1)).0"
            ;;
        patch)
            echo "${major}.${minor}.$((patch + 1))"
            ;;
    esac
}

# Default values
INCREMENT_TYPE=""
VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--patch)
            INCREMENT_TYPE="patch"
            shift
            ;;
        -m|--minor)
            INCREMENT_TYPE="minor"
            shift
            ;;
        -M|--major)
            INCREMENT_TYPE="major"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            VERSION="$1"
            shift
            ;;
    esac
done

# Show help if no flag and no version provided
if [ -z "$INCREMENT_TYPE" ] && [ -z "$VERSION" ]; then
    show_help
    exit 0
fi

# Get or calculate version
if [ -z "$VERSION" ]; then
    CURRENT_VERSION=$(get_current_version)
    VERSION=$(increment_version "$CURRENT_VERSION" "$INCREMENT_TYPE")
    echo "Current version: $CURRENT_VERSION"
    echo "New version: $VERSION ($INCREMENT_TYPE increment)"
else
    echo "Using explicit version: $VERSION"
fi

DATE=$(date +%Y-%m-%d)

# Validate version format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z"
    exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --staged --quiet; then
    echo "Error: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

echo "Releasing version $VERSION..."

# Update pyproject.toml
sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
echo "Updated pyproject.toml"

# Update frame_detection/__init__.py
sed -i '' "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" frame_detection/__init__.py
echo "Updated frame_detection/__init__.py"

# Update CHANGELOG.md: rename [Unreleased] to [VERSION] and add new [Unreleased] section
sed -i '' "s/^## \[Unreleased\]/## [Unreleased]\n\n## [$VERSION] - $DATE/" CHANGELOG.md
echo "Updated CHANGELOG.md"

# Commit changes
git add pyproject.toml frame_detection/__init__.py CHANGELOG.md
git commit -m "Release v$VERSION"
echo "Committed changes"

# Create and push tag
git tag "v$VERSION"
git push origin main
git push origin "v$VERSION"

echo ""
echo "Released v$VERSION"
echo "GitHub Actions will now build and publish the release."
