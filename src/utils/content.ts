import { getCollection, type CollectionEntry } from "astro:content";
import { getAssetPath } from "./url";
import { slugify } from "./text";

import { POSTS_PATH, PAGES_PATH, NOTES_PATH } from "../content.config";
export type Post = CollectionEntry<"posts">;
export type Page = CollectionEntry<"pages">;
export type Note = CollectionEntry<"notes">;

export type AnyEntry = {
  id: string;
  published: Date;
  title: string;
  href: string;
  collection: 'posts' | 'notes';
  category?: string;
  tags?: string[];
  color?: string;
};

export type AnyCollectionEntry = Post | Note;
export type TranslatableEntry = Post | Note | Page;
export type ContentLanguage = TranslatableEntry["data"]["lang"];

export type TranslationLink = {
  lang: ContentLanguage;
  href: string;
};

let postsCache: Post[] | null = null;
let pagesCache: Page[] | null = null;
let notesCache: Note[] | null = null;

function isVisiblePost(post: Post): boolean {
  // Show everything in development
  if (import.meta.env.DEV) {
    return true;
  }

  const isDraft = post.data.draft;

  const isFuturePost =
    new Date(post.data.published).getTime() >
    Date.now();

  return !isDraft && !isFuturePost;
}

function sortPosts(posts: Post[]): Post[] {
  return posts.sort(
    (a, b) => b.data.published.valueOf() - a.data.published.valueOf()
  );
}

export async function getAllPosts(): Promise<Post[]> {
  if (postsCache) {
    return postsCache;
  }

  const posts = await getCollection(
    "posts",
    isVisiblePost
  );

  postsCache = sortPosts(posts);

  return postsCache;
}

// ── Pages ──────────────────────────────────────────────────────────────────────

function sortPages(pages: Page[]): Page[] {
  return pages.sort((a, b) => {
    const aDate = new Date(a.data.updated ?? 0).getTime();
    const bDate = new Date(b.data.updated ?? 0).getTime();
    return bDate - aDate;
  });
}

export async function getAllPages(): Promise<Page[]> {
  if (pagesCache) {
    return pagesCache;
  }

  const pages = await getCollection("pages");

  pagesCache = sortPages(pages);

  return pagesCache;
}

// ── Notes ──────────────────────────────────────────────────────────────────────

function isVisibleNote(note: Note): boolean {
  if (import.meta.env.DEV) {
    return true;
  }

  const isDraft = note.data.draft;

  const isFutureNote =
    new Date(note.data.published).getTime() > Date.now();

  return !isDraft && !isFutureNote;
}

function sortNotes(notes: Note[]): Note[] {
  return notes.sort(
    (a, b) => b.data.published.valueOf() - a.data.published.valueOf()
  );
}

export async function getAllNotes(): Promise<Note[]> {
  if (notesCache) {
    return notesCache;
  }

  const notes = await getCollection("notes", isVisibleNote);

  notesCache = sortNotes(notes);

  return notesCache;
}

/**
 * Posts and notes combined, sorted by published desc. The one shared source
 * of "every entry" for the archive and browse pages.
 */
export async function getAllEntries(): Promise<(Post | Note)[]> {
  const [posts, notes] = await Promise.all([getAllPosts(), getAllNotes()]);

  return [...posts, ...notes].sort(
    (a, b) => b.data.published.valueOf() - a.data.published.valueOf()
  );
}

export async function getPrimaryPosts(): Promise<Post[]> {
  return getPrimaryTranslations(await getAllPosts());
}

export async function getPrimaryNotes(): Promise<Note[]> {
  return getPrimaryTranslations(await getAllNotes());
}

export async function getPrimaryEntries(): Promise<(Post | Note)[]> {
  const [posts, notes] = await Promise.all([getPrimaryPosts(), getPrimaryNotes()]);

  return [...posts, ...notes].sort(
    (a, b) => b.data.published.valueOf() - a.data.published.valueOf()
  );
}

/**
 * Remove hidden folders and normalize directory segments.
 *
 * Example:
 * posts/_2026/Japan Beyond Places.md
 * -> []
 *
 * posts/travel/Japan/Tokyo.md
 * -> ["travel", "japan"]
 */
export function getPostPathSegments(
  filePath?: string
): string[] {
  if (!filePath) {
    return [];
  }

  return filePath
    .replace(POSTS_PATH, "")
    .split("/")
    .filter(Boolean)
    .filter((segment) => !segment.startsWith("_"))
    .slice(0, -1)
    .map(slugify);
}

/**
 * Get the final slug segment from Astro content entry ID.
 *
 * Example:
 * "travel/tokyo-beyond-places"
 * -> "tokyo-beyond-places"
 */
export function getPostSlugSegment(id: string): string {
  const segments = id.split("/");

  return segments.at(-1) ?? id;
}

/**
 * Generate nested slug path from file structure.
 *
 * Example:
 * travel/japan/tokyo.md
 * -> "travel/japan/tokyo"
 */
export function getPostSlugPath(
  id: string,
  filePath?: string
): string {
  const segments = getPostPathSegments(filePath);

  const slug =
    slugify(getPostSlugSegment(id));

  return segments.length > 0
    ? [...segments, slug].join("/")
    : slug;
}

/**
 * Route param slug used in getStaticPaths().
 *
 * Example:
 * "/travel/japan/tokyo"
 */
export function getPostSlug(
  id: string,
  filePath?: string
): string {
  return `/${getPostSlugPath(id, filePath)}`;
}

export function getPagePathSegments(
  filePath?: string
): string[] {
  if (!filePath) {
    return [];
  }

  return filePath
    .replace(PAGES_PATH, "")
    .split("/")
    .filter(Boolean)
    .filter((segment) => !segment.startsWith("_"))
    .slice(0, -1)
    .map(slugify);
}

export function getPageSlugPath(
  id: string,
  filePath?: string
): string {
  const segments = getPagePathSegments(filePath);
  const slug = slugify(getPostSlugSegment(id));

  return segments.length > 0
    ? [...segments, slug].join("/")
    : slug;
}

export function getPageSlug(
  id: string,
  filePath?: string
): string {
  return `/${getPageSlugPath(id, filePath)}`;
}

/**
 * Full page URL — pages are served at the root (no /pages/ prefix).
 *
 * Example:
 * "about" -> "/about"
 */
export function getPageUrl(
  id: string,
  filePath?: string
): string {
  return getAssetPath(getPageSlugPath(id, filePath));
}

/**
 * Full post URL.
 *
 * Example:
 * "/posts/travel/japan/tokyo"
 */
export function getPostUrl(
  id: string,
  filePath?: string
): string {
  return getAssetPath(
    `posts/${getPostSlugPath(id, filePath)}`
  );
}

/**
 * Remove hidden folders and normalize directory segments for notes.
 *
 * Example:
 * notes/_drafts/on-attention.md
 * -> []
 *
 * notes/tech/self-hosting.md
 * -> ["tech"]
 */
export function getNotePathSegments(
  filePath?: string
): string[] {
  if (!filePath) {
    return [];
  }

  return filePath
    .replace(NOTES_PATH, "")
    .split("/")
    .filter(Boolean)
    .filter((segment) => !segment.startsWith("_"))
    .slice(0, -1)
    .map(slugify);
}

/**
 * Generate nested slug path from note file structure.
 *
 * Example:
 * tech/self-hosting.md
 * -> "tech/self-hosting"
 */
export function getNoteSlugPath(
  id: string,
  filePath?: string
): string {
  const segments = getNotePathSegments(filePath);
  const slug = slugify(getPostSlugSegment(id));

  return segments.length > 0
    ? [...segments, slug].join("/")
    : slug;
}

/**
 * Route param slug used in getStaticPaths() for notes.
 *
 * Example:
 * "/tech/self-hosting"
 */
export function getNoteSlug(
  id: string,
  filePath?: string
): string {
  return `/${getNoteSlugPath(id, filePath)}`;
}

/**
 * Full note URL.
 *
 * Example:
 * "/notes/tech/self-hosting"
 */
export function getNoteUrl(
  id: string,
  filePath?: string
): string {
  return getAssetPath(
    `notes/${getNoteSlugPath(id, filePath)}`
  );
}

export function getEntryUrl(entry: TranslatableEntry): string {
  if (entry.collection === "posts") {
    return getPostUrl(entry.id, entry.filePath);
  }

  if (entry.collection === "notes") {
    return getNoteUrl(entry.id, entry.filePath);
  }

  return getPageUrl(entry.id, entry.filePath);
}

// ── Translations ───────────────────────────────────────────────────────────────

function compareTranslations<T extends TranslatableEntry>(a: T, b: T): number {
  if (a.data.lang === b.data.lang) return a.id.localeCompare(b.id);
  if (a.data.lang === "en") return -1;
  if (b.data.lang === "en") return 1;
  return a.data.lang.localeCompare(b.data.lang);
}

function getTranslationGroupKey(entry: TranslatableEntry): string {
  return entry.data.translationKey ?? `${entry.collection}:${entry.id}`;
}

export function groupTranslationVariants<T extends TranslatableEntry>(
  entries: T[]
): Map<string, T[]> {
  const groups = new Map<string, T[]>();

  for (const entry of entries) {
    const key = getTranslationGroupKey(entry);
    const group = groups.get(key) ?? [];
    group.push(entry);
    groups.set(key, group);
  }

  for (const group of groups.values()) {
    group.sort(compareTranslations);
  }

  return groups;
}

export function findTranslationVariants<T extends TranslatableEntry>(
  entry: T,
  entries: T[]
): T[] {
  return groupTranslationVariants(entries).get(getTranslationGroupKey(entry)) ?? [];
}

export function getPrimaryTranslation<T extends TranslatableEntry>(
  variants: T[]
): T | undefined {
  return [...variants].sort(compareTranslations)[0];
}

export function getPrimaryTranslations<T extends TranslatableEntry>(entries: T[]): T[] {
  return [...groupTranslationVariants(entries).values()]
    .map(getPrimaryTranslation)
    .filter((entry): entry is T => entry !== undefined);
}

export function getTranslationLinks<T extends TranslatableEntry>(
  entry: T,
  entries: T[]
): TranslationLink[] {
  const variants = findTranslationVariants(entry, entries);
  if (variants.length < 2) return [];

  return variants.map((variant) => ({
    lang: variant.data.lang,
    href: getEntryUrl(variant),
  }));
}

export async function buildBacklinkMap(): Promise<Map<string, { title: string; slug: string }[]>> {
  const notes = await getAllNotes();
  const map = new Map<string, { title: string; slug: string }[]>();

  for (const note of notes) {
    const body = note.body ?? '';
    const noteSlug = getNoteSlugPath(note.id, note.filePath);
    const entry = { title: note.data.title, slug: noteSlug };

    // wikilinks: [[Target]], [[Target|Alias]], [[Target#Heading]] — key by the
    // same slug wikilinkResolver (src/plugins/satteri.ts) resolves the target to.
    for (const match of body.matchAll(/\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)) {
      const target = match[1].split('#')[0].trim();
      if (!target) continue;
      const key = `${slugify(target)}/${note.data.lang}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(entry);
    }

    // markdown links: [text](/notes/slug)
    for (const match of body.matchAll(/\[([^\]]+)\]\(\/notes\/([^)#]+)/g)) {
      const linkedSlug = match[2];
      const key = /\/(?:en|vi)$/.test(linkedSlug)
        ? linkedSlug
        : `${linkedSlug}/${note.data.lang}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(entry);
    }
  }

  return map;
}

export async function getSeriesArticles(series: string) {
  const posts = await getAllPosts();
  return posts
    .filter(p => p.data.series === series)
    .sort((a, b) => (a.data.seriesOrder ?? 0) - (b.data.seriesOrder ?? 0));
}

// ── Grouping ───────────────────────────────────────────────────────────────────

export function getPostsGroupedByYear(
  entries: Post[]
): [string, Post[]][] {
  const grouped = entries.reduce<Record<string, Post[]>>((acc, entry) => {
    const year = entry.data.published.getFullYear().toString();
    (acc[year] ??= []).push(entry);
    return acc;
  }, {});

  for (const year in grouped) {
    grouped[year].sort(
      (a, b) => b.data.published.valueOf() - a.data.published.valueOf()
    );
  }

  return Object.entries(grouped).sort(
    ([a], [b]) => Number(b) - Number(a)
  );
}

export function getNotesGroupedByYear(
  entries: Note[]
): [string, Note[]][] {
  const grouped = entries.reduce<Record<string, Note[]>>((acc, entry) => {
    const year = entry.data.published.getFullYear().toString();
    (acc[year] ??= []).push(entry);
    return acc;
  }, {});

  for (const year in grouped) {
    grouped[year].sort(
      (a, b) => b.data.published.valueOf() - a.data.published.valueOf()
    );
  }

  return Object.entries(grouped).sort(
    ([a], [b]) => Number(b) - Number(a)
  );
}
