---
layout: default
title: "Build with Rust & Solana"
description: "Deep-dive into Rust and Solana engineering with curated tutorials and production-ready patterns."
---

<section class="hero">
  <span class="hero-eyebrow">Curated technical content</span>
  <h1 class="hero-title">Master Rust & High-Performance Solana Systems</h1>
  <p class="hero-lede">Systems-grade programming for builders shipping on Solana. Explore hands-on guides, architectural deep dives, and production-ready patterns.</p>
  <div class="hero-actions">
    <a class="primary" href="{{ '/rust-solana/' | relative_url }}">Explore Rust &amp; Solana Lab</a>
  </div>
</section>

<section aria-label="Topics">
  <div class="topic-grid single-topic">
    <article class="topic-card featured">
      <span class="topic-icon" role="img" aria-label="Rust and Solana">🦀⚡</span>
      <h3>Rust &amp; Solana Engineering</h3>
      <p>Systems-grade code for builders shipping on Solana: Rust generics, account safety, program patterns, testing flows, and deployment pipelines.</p>
      <div class="topic-highlights">
        <span class="highlight">Solana Programs</span>
        <span class="highlight">On-chain Safety</span>
        <span class="highlight">Performance Tuning</span>
        <span class="highlight">Rust Generics</span>
        <span class="highlight">Testing Patterns</span>
      </div>
      <a class="topic-button" href="{{ '/rust-solana/' | relative_url }}">View Rust &amp; Solana Posts →</a>
    </article>
  </div>
</section>

<style>
.topic-grid.single-topic {
  display: flex;
  justify-content: center;
  max-width: 600px;
  margin: 0 auto;
}

.topic-card.featured {
  max-width: 100%;
  transform: scale(1.05);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.hero-actions {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 2rem;
}

.hero-actions .primary {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  text-decoration: none;
  font-weight: 600;
  transition: transform 0.2s ease;
}

.hero-actions .primary:hover {
  transform: translateY(-2px);
}
</style>