/* global process */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __filename = fileURLToPath(import.meta.url)

export const rootDir = path.resolve(path.dirname(__filename), '..')
export const docsDir = path.join(rootDir, 'docs')
export const distDir = path.join(docsDir, '.vitepress', 'dist')
export const configPath = path.join(docsDir, '.vitepress', 'config.mjs')
export const packageJsonPath = path.join(rootDir, 'package.json')
export const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
export const logoPath = '../public/readme/readmelogo.png'

// --- Book metadata ---

const sanbuGithubUrl =
  process.env.PDF_SANBU_GITHUB_URL || 'https://github.com/sanbuphy'

export const bookVersion =
  process.env.PDF_VERSION || '0.1.0'

export const bookAuthor =
  process.env.PDF_AUTHORS ||
  process.env.PDF_AUTHOR ||
  packageJson.author ||
  `letslego;  (${sanbuGithubUrl})`

export const bookTitle = '——'

export const bookTagline =
  '：、LLM 、RLVR '

export const bookLicense = 'CC-BY-NC-SA-4.0'

export const bookBuildDate =
  process.env.PDF_BUILD_DATE ||
  new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'long',
    timeZone: 'Asia/Shanghai'
  }).format(new Date())

export function repositoryUrl() {
  if (process.env.PDF_GITHUB_URL) return process.env.PDF_GITHUB_URL

  const githubRepository = process.env.GITHUB_REPOSITORY
  if (githubRepository && githubRepository.includes('/')) {
    return `https://github.com/${githubRepository.replace(/\.git$/, '')}`
  }

  const repository =
    typeof packageJson.repository === 'string'
      ? packageJson.repository
      : packageJson.repository?.url || ''

  const sshMatch = repository.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
  if (sshMatch) {
    return `https://github.com/${sshMatch[1]}/${sshMatch[2]}`
  }

  const normalized = repository
    .replace(/^git\+/, '')
    .replace(/^https?:\/\/github\.com\//, '')
    .replace(/^git@github\.com:/, '')
    .replace(/\.git$/, '')

  if (normalized.includes('/')) {
    return `https://github.com/${normalized}`
  }

  return 'https://github.com/letslego/hands-on-modern-rl'
}

export const bookGithubUrl = repositoryUrl()

// --- Utility functions ---

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

export function stripFrontmatter(value) {
  if (!value.startsWith('---\n')) return value
  const end = value.indexOf('\n---\n', 4)
  if (end === -1) return value
  return value.slice(end + 5)
}

export function stripScriptSetup(value) {
  return value.replace(/^\s*<script\b[\s\S]*?<\/script>\s*/i, '')
}

export function flattenSidebar(items, pages = [], seen = new Set()) {
  for (const item of items || []) {
    if (item.link && !item.link.includes('#') && !seen.has(item.link)) {
      seen.add(item.link)
      pages.push({ title: item.text || item.link, link: item.link })
    }
    if (item.items) {
      flattenSidebar(item.items, pages, seen)
    }
  }
  return pages
}

export function splitSuffix(value) {
  const hashIndex = value.indexOf('#')
  const queryIndex = value.indexOf('?')
  const suffixIndexes = [hashIndex, queryIndex].filter((index) => index >= 0)
  const suffixIndex = suffixIndexes.length ? Math.min(...suffixIndexes) : -1

  if (suffixIndex === -1) {
    return { clean: value, suffix: '' }
  }

  return {
    clean: value.slice(0, suffixIndex),
    suffix: value.slice(suffixIndex)
  }
}

export function pagePathForLink(link) {
  const { clean } = splitSuffix(link.replace(/^\//, ''))
  const normalized = clean.replace(/\/$/, '')
  const candidates = [
    path.join(docsDir, `${normalized}.md`),
    path.join(docsDir, normalized, 'index.md')
  ]

  return candidates.find((candidate) => fs.existsSync(candidate)) || null
}

export async function loadVitePressConfig() {
  const module = await import(pathToFileURL(configPath).href)
  return module.default
}

export function loadSidebar(config) {
  return (
    config.locales?.zh?.themeConfig?.sidebar?.['/'] ||
    config.themeConfig?.sidebar?.['/'] ||
    []
  )
}

// --- Front matter content (structured data) ---

export const frontMatter = {
  cover: {
    kicker: 'Open Textbook · Modern Reinforcement Learning',
    titleEn: 'Hands-on Modern RL',
    titleZh: bookTitle,
    description: `${bookTagline}。、， MDP、、PPO、DPO、GRPO、RLVR、Agentic RL 。`,
    publisher: 'letslego Open Course Series'
  },
  intro: {
    title: '',
    paragraphs: [
      '<strong>Hands-on Modern RL</strong> 。 PDF，、、。',
      '""：、、，、，、、、、。',
      ' CartPole、DQN、Actor-Critic  PPO  LLM ， RLHF、DPO、GRPO、RLVR、、Deep Research、VLM 、。'
    ],
    audience: {
      title: '',
      items: [
        '；',
        '、LLM  Agentic RL ；',
        ' RLHF、DPO、GRPO、RLVR ；',
        '、，。'
      ]
    },
    howToRead: {
      title: '',
      paragraphs: [
        '，； RL， PPO、DPO、GRPO  Agentic RL ，。'
      ]
    }
  },
  publicationNote: {
    title: '',
    iterationNote:
      '。v0.1.0 、、、PDF 、；、、、。',
    disclaimer:
      '， GitHub 。 issue、pull request 、。',
    licenseSection: {
      title: '',
      text: '，、。，、、 GitHub 。',
    }
  }
}
