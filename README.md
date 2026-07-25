# The Lazy Blog

The personal publishing site of [Lazy Yuuki](https://github.com/LazyYuuki),
focused on software engineering, artificial intelligence, startups, and lessons
learned while building things.

Live site: [lazyyuuki.github.io](https://lazyyuuki.github.io)

The site uses Astro and Patrika's notebook-inspired visual language, with an
emphasis on typography, whitespace, and long-form reading.

## Content

The writing is organized into a few distinct editorial types:

- **Articles** are longer technical essays and personal reflections about AI,
  engineering, writing, startups, and building. They live in
  [`src/content/posts/`](src/content/posts/).
- **Quotes** collect passages from books worth remembering. They live in
  [`src/content/pages/quotes/`](src/content/pages/quotes/) and are published
  under `/quotes`.
- **Lists** track books, project ideas, ambitions, problems worth solving, and
  words to live by. They live in
  [`src/content/pages/lists/`](src/content/pages/lists/) and are published under
  `/lists`.
- **Notes** are provisional working notes and braindumps. They live in
  [`src/content/notes/`](src/content/notes/). Draft notes are available during
  local development but excluded from production builds.
- **Pages** contain standalone material such as the About page and the Quotes
  and Lists indexes. They live in [`src/content/pages/`](src/content/pages/).

All dated entries are shown newest first. Draft and future-dated posts are not
published in production.

## Features

- Astro content collections with validated frontmatter
- Obsidian-style wikilinks, image embeds, highlights, and annotations
- Syntax highlighting and Markdown footnotes
- Pagefind full-text search
- Light and dark themes with a persisted preference
- Generated Open Graph images, RSS, and sitemap
- Static deployment to GitHub Pages

## Development

Requires Node.js 22.12 or newer and pnpm 10.14.0.

```sh
corepack pnpm install
corepack pnpm dev
```

The development server runs at `http://localhost:4321`.

| Command | Action |
| :--- | :--- |
| `corepack pnpm dev` | Start the Astro development server in the background |
| `corepack pnpm build` | Build the production site and Pagefind index |
| `corepack pnpm preview` | Preview the production build locally |
| `corepack pnpm astro dev stop` | Stop the background development server |

## Configuration

Site identity, navigation, author details, and content settings are configured
in [`src/content/siteConfig/config.yaml`](src/content/siteConfig/config.yaml).
The canonical deployment URL is also set in
[`astro.config.mjs`](astro.config.mjs).

Pushes to `main` are built and deployed to GitHub Pages through
[`.github/workflows/deploy.yaml`](.github/workflows/deploy.yaml).
