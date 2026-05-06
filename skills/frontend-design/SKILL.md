---
name: frontend-design
description: Build HTML/CSS/JS interfaces with production-quality, modern aesthetics
tags: html, css, javascript, ui, web
---

# Frontend Design Skill

## When to load this skill
Load this skill when the mission involves:
- Creating HTML pages, landing pages, dashboards, or web components
- Writing CSS stylesheets or Tailwind utility classes
- Building interactive UI with vanilla JS or a framework
- Producing any file that will be opened in a browser

---

## Design Principles

### Typography
- Use system font stacks: `font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`
- Body: 16px / line-height 1.6. Headings: tight line-height 1.1–1.2.
- Limit to 2 typefaces maximum. Never use Comic Sans, Papyrus, or Impact.

### Colour
- Start from a single brand colour and derive the palette algorithmically (HSL shifts).
- Dark mode first: background `#0f0f0f`, surface `#1a1a1a`, border `#2a2a2a`.
- Text: primary `#f0f0f0`, secondary `#a0a0a0`, muted `#606060`.
- Accent: one warm colour (e.g. `#f5a623` amber or `#7c6af7` violet). No rainbow.

### Spacing
- Use an 8px grid. Margins/paddings: 8, 16, 24, 32, 48, 64, 96.
- Generous whitespace. Content width max 720px for prose, 1200px for dashboards.

### Components
```css
/* Card */
.card {
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  border-radius: 12px;
  padding: 24px;
  transition: border-color 0.2s;
}
.card:hover { border-color: #444; }

/* Button — primary */
.btn {
  background: #f5a623;
  color: #0f0f0f;
  border: none;
  border-radius: 8px;
  padding: 10px 20px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.15s;
}
.btn:hover { opacity: 0.85; }

/* Input */
.input {
  background: #111;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 10px 14px;
  color: #f0f0f0;
  outline: none;
  transition: border-color 0.2s;
}
.input:focus { border-color: #f5a623; }
```

### File output rules
- Single self-contained `.html` file unless the user specifies otherwise.
- Inline all CSS in `<style>` tags. Inline JS at bottom of `<body>`.
- No external CDN dependencies unless explicitly requested.
- Always include `<meta name="viewport" content="width=device-width, initial-scale=1">`.
- Validate: no unclosed tags, no missing alt attributes on images.

### Quality checklist before writing the file
- [ ] Contrast ratio ≥ 4.5:1 for body text
- [ ] Mobile viewport handled
- [ ] No hardcoded pixel font sizes (use rem/em)
- [ ] Hover/focus states on all interactive elements
- [ ] Semantic HTML (nav, main, section, article, footer)