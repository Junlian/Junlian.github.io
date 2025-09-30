# Junlian's Tech Blog

A Jekyll-powered blog focused on Rust programming, Solana blockchain development, and modern web technologies.

## 🚀 Quick Start: Adding a New Blog Post

Use the automated script to add new blog posts in seconds:

```bash
./add_blog_post.sh "Your Blog Title" "/path/to/your/markdown/file.md" "tag1,tag2,tag3"
```

### Examples:

```bash
# Basic usage
./add_blog_post.sh "Advanced Rust Patterns" "/Users/macv3/my_posts/rust_patterns.md"

# With custom tags
./add_blog_post.sh "Solana Smart Contracts" "/path/to/solana_post.md" "solana,smart-contracts,web3"

# From external directory
./add_blog_post.sh "Blockchain Security" "/Users/macv3/course_root/security_post.md" "blockchain,security,rust"
```

### What the script does:
1. ✅ Reads your markdown file
2. ✅ Adds proper Jekyll front matter
3. ✅ Uses current date (no future date issues!)
4. ✅ Generates SEO-friendly filename
5. ✅ Commits and pushes to GitHub
6. ✅ Triggers automatic GitHub Pages deployment

## 📁 Project Structure

```
├── _posts/                 # Blog posts (YYYY-MM-DD-title.md)
├── _layouts/               # Jekyll layouts
├── _includes/              # Reusable components
├── pages/                  # Static pages
├── add_blog_post.sh       # 🎯 Quick blog post script
└── _config.yml            # Jekyll configuration
```

## 🌐 Live Website

- **Homepage**: https://junlian.github.io
- **GitHub Repository**: https://github.com/Junlian/Junlian.github.io
- **Build Status**: https://github.com/Junlian/Junlian.github.io/actions

## 🛠️ Manual Development

If you need to work manually:

```bash
# Install dependencies
bundle install

# Serve locally (optional)
bundle exec jekyll serve

# Manual git workflow
git add .
git commit -m "Your commit message"
git push origin main
```

## 📝 Blog Post Format

The script automatically creates proper Jekyll front matter:

```yaml
---
layout: post
title: "Your Title"
date: 2025-09-30
author: "Junlian"
description: "SEO description"
excerpt: "Preview text..."
tags: [rust, blockchain, programming]
categories: [Programming, Blockchain]
---

Your content here...
```

## 🎯 Pro Tips

- **Always use the script** - it handles dates, formatting, and deployment automatically
- **Keep source files organized** - store your drafts in a separate folder
- **Use descriptive titles** - they become the URL slug
- **Add relevant tags** - helps with SEO and organization

---

*Built with Jekyll • Hosted on GitHub Pages • Optimized for SEO*