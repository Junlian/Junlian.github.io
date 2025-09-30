#!/bin/bash

# Simple script to add a new blog post to Jekyll site
# Usage: ./add_blog_post.sh "Blog Title" "path/to/source/file.md" [tags]

set -e

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"Blog Title\" \"path/to/source/file.md\" [\"tag1,tag2,tag3\"]"
    echo "Example: $0 \"My New Post\" \"/path/to/post.md\" \"rust,blockchain,tutorial\""
    exit 1
fi

TITLE="$1"
SOURCE_FILE="$2"
TAGS="${3:-rust,blockchain,programming}"

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file '$SOURCE_FILE' not found!"
    exit 1
fi

# Generate filename-friendly title
FILENAME_TITLE=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g')

# Get current date
CURRENT_DATE=$(date +%Y-%m-%d)

# Create the new blog post filename
NEW_POST_FILE="_posts/${CURRENT_DATE}-${FILENAME_TITLE}.md"

echo "Creating new blog post: $NEW_POST_FILE"

# Read the source file content
SOURCE_CONTENT=$(cat "$SOURCE_FILE")

# Create the Jekyll front matter and content
cat > "$NEW_POST_FILE" << EOF
---
layout: post
title: "$TITLE"
date: $CURRENT_DATE
author: "Junlian"
description: "$(echo "$TITLE" | head -c 150)"
excerpt: "$(echo "$SOURCE_CONTENT" | head -n 5 | tail -n 1 | head -c 150)..."
tags: [$(echo "$TAGS" | sed 's/,/, /g')]
categories: [Programming, Blockchain]
---

$SOURCE_CONTENT
EOF

echo "✅ Blog post created: $NEW_POST_FILE"

# Add to git
git add "$NEW_POST_FILE"
echo "✅ Added to git staging"

# Commit
git commit -m "Add new blog post: $TITLE"
echo "✅ Committed to git"

# Push to GitHub
git push origin main
echo "✅ Pushed to GitHub"

echo ""
echo "🎉 Blog post successfully published!"
echo "📝 File: $NEW_POST_FILE"
echo "🌐 Will be available at: https://junlian.github.io/$(echo "$FILENAME_TITLE" | sed 's/-$//')/"
echo ""
echo "⏱️  GitHub Pages will rebuild in 1-2 minutes."
echo "🔄 You can check the status at: https://github.com/Junlian/Junlian.github.io/actions"