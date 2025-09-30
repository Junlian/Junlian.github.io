---
layout: default
title: "Junlian's Tech Blog - Rust, Solana & Web Development"
description: "Welcome to Junlian's technical blog covering Rust programming, Solana blockchain development, and modern web technologies. Discover tutorials, insights, and deals on cutting-edge tech."
---

# Welcome to Junlian's Tech Blog

## Latest Posts

{% for post in site.posts limit:5 %}
### [{{ post.title }}]({{ post.url }})
*{{ post.date | date: "%B %d, %Y" }}*

{{ post.excerpt }}

[Read more →]({{ post.url }})

---
{% endfor %}

## About This Blog

Welcome to my technical blog where I share insights about:

- **Rust Programming** - Systems programming, performance optimization, and best practices
- **Solana Blockchain** - Smart contract development, DeFi protocols, and Web3 technologies  
- **Web Development** - Modern frameworks, SEO optimization, and developer tools
- **Tech Deals** - Curated recommendations for developers and tech enthusiasts

## Featured Content

- 🦀 [Introduction to Rust and Solana Development]({% post_url 2025-09-30-rust-solana-intro %})
- 💰 [Tech Deals & Recommendations](/pages/deals.html)

## Connect With Me

Stay updated with the latest posts and tech insights. This blog is optimized for search engines and designed to help developers learn and grow.

---

*Built with Jekyll and hosted on GitHub Pages. Optimized for performance and SEO.*