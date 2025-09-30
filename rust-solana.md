---
layout: default
title: "Rust & Solana Engineering"
description: "Deep dives into Solana program architecture, Rust systems patterns, and production deployment checklists."
permalink: /rust-solana/
---

<section class="hero">
  <span class="hero-eyebrow">High-performance web3</span>
  <h1 class="hero-title">Ship Reliable Programs on Solana with Modern Rust</h1>
  <p class="hero-lede">Practical notes on architecting Solana programs, safeguarding accounts, optimizing execution, and deploying at scale with Rust.</p>
  <div class="hero-actions">
    <a class="primary" href="#rust-solana-posts">Browse build guides</a>
    <a class="ghost" href="{{ '/' | relative_url }}">Return home</a>
  </div>
</section>

<section id="rust-solana-posts" aria-label="Rust and Solana Articles">
  {% assign ordered_posts = site.posts | sort: 'date' | reverse %}
  {% assign rust_count = 0 %}
  <div class="post-list">
    {% for post in ordered_posts %}
      {% assign matches = 0 %}
      {% if post.tags contains 'rust' or post.tags contains 'solana' or post.tags contains 'blockchain' or post.tags contains 'smart-contracts' or post.tags contains 'programming' or post.categories contains 'rust' or post.categories contains 'solana' or post.categories contains 'blockchain' or post.categories contains 'Programming' or post.categories contains 'Blockchain' or post.url contains 'rust' or post.url contains 'solana' %}
        {% assign matches = 1 %}
      {% endif %}
      {% if matches == 1 %}
        {% assign rust_count = rust_count | plus: 1 %}
        <article class="post-card">
          <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
          <p class="post-meta">{{ post.date | date: "%B %d, %Y" }}</p>
          <p>{{ post.excerpt | strip_html | truncatewords: 32 }}</p>
          {% if post.tags %}
            {% assign top_tags = post.tags | slice: 0, 3 %}
            {% if top_tags.size > 0 %}
              <div class="post-tags">
                {% for tag in top_tags %}
                  <span class="tag-chip">#{{ tag }}</span>
                {% endfor %}
              </div>
            {% endif %}
          {% endif %}
        </article>
      {% endif %}
    {% endfor %}
  </div>
  {% if rust_count == 0 %}
    <p>No Rust or Solana articles have been published yet. Stay tuned for hands-on walkthroughs and deployment notes.</p>
  {% endif %}
</section>
