#!/bin/bash
set -e

# Release script: updates versions, commits, tags, and pushes

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.0.3"
    exit 1
fi

VERSION="$1"
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
