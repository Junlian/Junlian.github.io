---
layout: default
title: "Junlian's Tech Blog - Rust, Solana & AI Development"
description: "Welcome to Junlian's technical blog covering Rust programming, Solana blockchain development, AI agents, and modern web technologies. Choose your learning path and discover cutting-edge tutorials."
---

# Welcome to Junlian's Tech Blog

## 🚀 Choose Your Learning Path

Explore our comprehensive content organized by main topics. Select the area that interests you most to dive deep into specialized tutorials and insights.

<div class="topic-selection">
  <div class="topic-card rust-solana">
    <div class="topic-icon">🦀⚡</div>
    <h3>Rust & Solana Blockchain Insights</h3>
    <p>Master systems programming with Rust and build high-performance blockchain applications on Solana. Learn smart contracts, DeFi protocols, and Web3 development.</p>
    <div class="topic-highlights">
      <span class="highlight">Smart Contracts</span>
      <span class="highlight">DeFi Development</span>
      <span class="highlight">Performance Optimization</span>
    </div>
    <a href="/rust-solana/" class="topic-button">Explore Rust & Solana →</a>
  </div>
  
  <div class="topic-card ai-agent">
    <div class="topic-icon">🤖🧠</div>
    <h3>AI Agent Development</h3>
    <p>Build intelligent systems and autonomous agents. Learn machine learning integration, natural language processing, and automation frameworks for modern AI applications.</p>
    <div class="topic-highlights">
      <span class="highlight">Intelligent Systems</span>
      <span class="highlight">Automation</span>
      <span class="highlight">ML Integration</span>
    </div>
    <a href="/ai-agent/" class="topic-button">Explore AI Agents →</a>
  </div>
</div>

<style>
.topic-selection {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2em;
  margin: 3em auto;
  max-width: 900px;
}

.topic-card {
  padding: 2em;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  text-align: center;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
  border: 1px solid #e9ecef;
}

.topic-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.rust-solana {
  border-left: 4px solid #3498db;
}

.ai-agent {
  border-left: 4px solid #e74c3c;
}

.topic-icon {
  font-size: 3em;
  margin-bottom: 0.5em;
}

.topic-card h3 {
  color: #2c3e50;
  margin-bottom: 1em;
  font-size: 1.4em;
}

.topic-card p {
  color: #555;
  line-height: 1.6;
  margin-bottom: 1.5em;
}

.topic-highlights {
  margin-bottom: 2em;
}

.highlight {
  display: inline-block;
  background-color: #ecf0f1;
  color: #2c3e50;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.85em;
  margin: 0.2em;
  font-weight: 500;
}

.rust-solana .highlight {
  background-color: #e3f2fd;
  color: #1976d2;
}

.ai-agent .highlight {
  background-color: #ffebee;
  color: #c62828;
}

.topic-button {
  display: inline-block;
  padding: 12px 24px;
  border-radius: 6px;
  text-decoration: none;
  font-weight: 600;
  transition: all 0.3s ease;
  color: white;
}

.rust-solana .topic-button {
  background-color: #3498db;
}

.rust-solana .topic-button:hover {
  background-color: #2980b9;
  text-decoration: none;
}

.ai-agent .topic-button {
  background-color: #e74c3c;
}

.ai-agent .topic-button:hover {
  background-color: #c0392b;
  text-decoration: none;
}

@media (max-width: 768px) {
  .topic-selection {
    grid-template-columns: 1fr;
  }
  
  .topic-card {
    padding: 1.5em;
  }
}
</style>