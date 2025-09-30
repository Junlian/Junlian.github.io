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

## 📚 Latest Posts Across All Topics

{% for post in site.posts limit:3 %}
<div class="recent-post">
  <h4><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h4>
  <div class="post-meta">{{ post.date | date: "%B %d, %Y" }}</div>
  <p>{{ post.excerpt | strip_html | truncatewords: 25 }}</p>
  <a href="{{ post.url | relative_url }}" class="read-more">Read Full Article →</a>
</div>
{% endfor %}

<div class="view-all-posts">
  <a href="#" onclick="showAllPosts()" class="view-all-button">View All Posts</a>
</div>

<div id="all-posts" class="all-posts-section" style="display: none;">
  <h3>All Blog Posts</h3>
  {% for post in site.posts offset:3 %}
  <div class="recent-post">
    <h4><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h4>
    <div class="post-meta">{{ post.date | date: "%B %d, %Y" }}</div>
    <p>{{ post.excerpt | strip_html | truncatewords: 20 }}</p>
    <a href="{{ post.url | relative_url }}" class="read-more">Read Full Article →</a>
  </div>
  {% endfor %}
</div>

## 🎯 What You'll Find Here

<div class="features-grid">
  <div class="feature-item">
    <h4>🔧 Hands-on Tutorials</h4>
    <p>Step-by-step guides with practical examples and real-world applications.</p>
  </div>
  
  <div class="feature-item">
    <h4>⚡ Performance Tips</h4>
    <p>Advanced optimization techniques for high-performance applications.</p>
  </div>
  
  <div class="feature-item">
    <h4>🚀 Latest Technologies</h4>
    <p>Stay updated with cutting-edge developments in Rust, Solana, and AI.</p>
  </div>
  
  <div class="feature-item">
    <h4>💰 Curated Resources</h4>
    <p>Handpicked tools, courses, and deals for developers and tech enthusiasts.</p>
  </div>
</div>

## 🔗 Quick Links

- 💰 [Tech Deals & Recommendations](/deals/) - Curated resources and special offers
- 🦀 [Rust & Solana Hub](/rust-solana/) - Blockchain development content
- 🤖 [AI Agent Hub](/ai-agent/) - Intelligent systems and automation

---

*Built with Jekyll and hosted on GitHub Pages. Optimized for performance and SEO. Updated: {{ site.time | date: "%Y-%m-%d" }}*

<style>
.topic-selection {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2em;
  margin: 3em 0;
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

.recent-post {
  margin-bottom: 2em;
  padding: 1.5em;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  background-color: #f8f9fa;
}

.recent-post h4 {
  margin-top: 0;
  margin-bottom: 0.5em;
}

.recent-post h4 a {
  color: #2c3e50;
  text-decoration: none;
}

.recent-post h4 a:hover {
  color: #3498db;
}

.post-meta {
  color: #6c757d;
  font-size: 0.9em;
  margin-bottom: 1em;
}

.read-more {
  color: #3498db;
  font-weight: 500;
  text-decoration: none;
}

.read-more:hover {
  text-decoration: underline;
}

.view-all-posts {
  text-align: center;
  margin: 2em 0;
}

.view-all-button {
  background-color: #6c757d;
  color: white;
  padding: 10px 20px;
  border-radius: 6px;
  text-decoration: none;
  font-weight: 500;
  cursor: pointer;
}

.view-all-button:hover {
  background-color: #5a6268;
  text-decoration: none;
}

.all-posts-section {
  margin-top: 2em;
  padding-top: 2em;
  border-top: 2px solid #e9ecef;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5em;
  margin: 3em 0;
}

.feature-item {
  padding: 1.5em;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  background-color: #ffffff;
  text-align: center;
}

.feature-item h4 {
  color: #2c3e50;
  margin-bottom: 1em;
}

@media (max-width: 768px) {
  .topic-selection {
    grid-template-columns: 1fr;
  }
  
  .topic-card {
    padding: 1.5em;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
  
  .recent-post {
    padding: 1em;
  }
}
</style>

<script>
function showAllPosts() {
  const allPostsSection = document.getElementById('all-posts');
  const viewAllButton = document.querySelector('.view-all-button');
  
  if (allPostsSection.style.display === 'none') {
    allPostsSection.style.display = 'block';
    viewAllButton.textContent = 'Hide Additional Posts';
  } else {
    allPostsSection.style.display = 'none';
    viewAllButton.textContent = 'View All Posts';
  }
}
</script>