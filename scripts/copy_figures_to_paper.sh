#!/bin/bash
# Script to copy relevant figures to the paper directory

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$PROJECT_ROOT/data/figures"
DEST_DIR="$PROJECT_ROOT/paper/6926abd82dda07b08e9c883a/latex/figures"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# List of figures referenced in the paper
FIGURES=(
    "adjusted_alpha_by_category.pdf"
    "adjusted_alpha_by_category_pangram.pdf"
    "adjusted_alpha_by_category_and_year.pdf"
    "adjusted_alpha_by_category_and_year_pangram.pdf"
    "adjusted_alpha_by_category_cs_subcategories.pdf"
    "adjusted_alpha_by_category_cs_subcategories_pangram.pdf"
    "adjusted_alpha_by_category_and_year_cs_subcategories.pdf"
    "adjusted_alpha_by_category_and_year_cs_subcategories_pangram.pdf"
)

echo "Copying figures to paper directory..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo ""

COPIED=0
MISSING=0

for figure in "${FIGURES[@]}"; do
    SOURCE_PATH="$SOURCE_DIR/$figure"
    DEST_PATH="$DEST_DIR/$figure"

    if [ -f "$SOURCE_PATH" ]; then
        cp "$SOURCE_PATH" "$DEST_PATH"
        echo "✓ Copied: $figure"
        ((COPIED++))
    else
        echo "✗ Missing: $figure"
        ((MISSING++))
    fi
done

echo ""
echo "Summary: $COPIED copied, $MISSING missing"

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "Warning: Some figures were not found in the source directory."
    echo "Make sure to generate them before compiling the paper."
else
    echo ""
    echo "All figures copied successfully!"
fi
