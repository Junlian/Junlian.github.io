---
layout: default
title: "Technical Development Hub"
description: "Master Rust & Solana development and AI agent systems with curated tutorials and production-ready patterns."
---

<section class="hero">
  <span class="hero-eyebrow">Curated technical content</span>
  <h1 class="hero-title">Master Modern Development</h1>
  <p class="hero-lede">Deep-dive into cutting-edge technologies: Rust & Solana blockchain development and intelligent AI agent systems. Explore hands-on guides, architectural patterns, and production-ready implementations.</p>
  <div class="hero-actions">
    <a class="primary" href="{{ '/rust-solana/' | relative_url }}">Rust &amp; Solana</a>
    <a class="secondary" href="{{ '/ai-agent/' | relative_url }}">AI Agents</a>
  </div>
</section>

<section aria-label="Topics">
  <div class="topic-grid">
    <article class="topic-card">
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
    
    <article class="topic-card">
      <span class="topic-icon" role="img" aria-label="AI Agents">🤖🧠</span>
      <h3>AI Agent Development</h3>
      <p>Build sophisticated AI agents with advanced memory systems, context management, and conversational interfaces for production applications.</p>
      <div class="topic-highlights">
        <span class="highlight">Memory Systems</span>
        <span class="highlight">Context Management</span>
        <span class="highlight">Prompt Engineering</span>
        <span class="highlight">Conversational AI</span>
        <span class="highlight">Performance Optimization</span>
      </div>
      <a class="topic-button" href="{{ '/ai-agent/' | relative_url }}">View AI Agent Posts →</a>
    </article>
  </div>
</section>

<style>
.topic-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2rem;
  max-width: 1000px;
  margin: 0 auto;
}

.topic-card {
  background: white;
  border: 1px solid #e9ecef;
  border-radius: 12px;
  padding: 2rem;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
}

.topic-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
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

.hero-actions .secondary {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  text-decoration: none;
  font-weight: 600;
  transition: transform 0.2s ease;
}

.hero-actions .primary:hover,
.hero-actions .secondary:hover {
  transform: translateY(-2px);
}

@media (max-width: 768px) {
  .topic-grid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
  
  .hero-actions {
    flex-direction: column;
    align-items: center;
  }
}
</style>