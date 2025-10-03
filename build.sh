#!/bin/bash

# Privium Website Build Script
# This script runs Pagefind to generate the search index
# Run this automatically on Render.com or manually before deployment

echo "Building Privium website search index..."

# Install Pagefind if not available
if ! command -v pagefind &> /dev/null; then
    echo "Installing Pagefind..."
    npm install -g pagefind
fi

# Run Pagefind to index the site (only index main 4 pages)
echo "Running Pagefind indexer..."
# Configuration is in pagefind.toml
npx -y pagefind --site . --glob "*.html"

echo "✓ Search index built successfully!"
echo "  - Indexed pages are in ./pagefind/"
echo "  - Only main site pages indexed (not client-portal or working)"
echo "  - Ready for deployment"
