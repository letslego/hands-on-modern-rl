/* global process */
import { defineConfig } from 'vitepress'
import { createLogger } from 'vite'
import { MermaidMarkdown } from 'vitepress-plugin-mermaid'
import { createRequire } from 'module'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const require = createRequire(import.meta.url)
const markdownItFootnote = require('markdown-it-footnote')
const markdownItContainer = require('markdown-it-container')
const katex = require('katex')

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const packageJsonPath = path.resolve(__dirname, '../../package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
const docsRoot = path.resolve(__dirname, '..')
const assetManifestPath = path.resolve(
  docsRoot,
  'public/optimized/asset-manifest.json'
)

function loadAssetManifest() {
  if (!fs.existsSync(assetManifestPath)) return { assets: {} }

  try {
    return JSON.parse(fs.readFileSync(assetManifestPath, 'utf8'))
  } catch {
    return { assets: {} }
  }
}

const assetManifest = loadAssetManifest()

const isVercel = process.env.VERCEL === '1' || !!process.env.VERCEL_URL

function parseRepository() {
  const repositoryUrl =
    process.env.GITHUB_REPOSITORY ||
    packageJson.repository?.url ||
    packageJson.repository ||
    ''

  if (repositoryUrl.includes('/')) {
    if (repositoryUrl.includes(':')) {
      const sshMatch = repositoryUrl.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
      if (sshMatch) {
        return { owner: sshMatch[1], repo: sshMatch[2] }
      }
    }

    const normalized = repositoryUrl
      .replace(/^https?:\/\/github\.com\//, '')
      .replace(/^git@github\.com:/, '')
      .replace(/\.git$/, '')

    if (normalized.includes('/')) {
      const [owner, repo] = normalized.split('/')
      return { owner, repo }
    }
  }

  return { owner: 'letslego', repo: packageJson.name || 'course-template' }
}

const { owner, repo } = parseRepository()
const base = process.env.BASE || (isVercel ? '/' : `/${repo}/`)
const siteUrl = process.env.SITE_URL || `https://${owner}.github.io/${repo}`
const editLinkPattern = `https://github.com/${owner}/${repo}/edit/main/docs/:path`
const enableLocalSearch = process.env.LOCAL_SEARCH !== '0'
const mermaidConfig = {
  securityLevel: 'loose',
  startOnLoad: false
}

function mermaidConfigPlugin() {
  const virtualModuleId = 'virtual:mermaid-config'
  const resolvedVirtualModuleId = `\0${virtualModuleId}`

  return {
    name: 'local-mermaid-config',
    resolveId(id) {
      if (id === virtualModuleId) {
        return resolvedVirtualModuleId
      }
    },
    load(id) {
      if (id === resolvedVirtualModuleId) {
        return `export default ${JSON.stringify(mermaidConfig)}`
      }
    }
  }
}

function katexDevPlugin() {
  return {
    name: 'katex-dev-inject',
    enforce: 'pre',
    transformIndexHtml(html) {
      const katexScripts = [
        `<script src="${base}katex.min.js"><\/script>`,
        `<script src="${base}auto-render.min.js"><\/script>`
      ]
      return html.replace('</head>', `${katexScripts.join('\n')}\n</head>`)
    }
  }
}

function normalizeBrokenDocPathPlugin() {
  const canonicalSegments = ['appendix_math', 'linear-algebra-basics']

  return {
    name: 'normalize-broken-doc-path',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next()

        const url = new URL(req.url, 'http://localhost')
        if (!url.pathname.includes('appendix')) return next()

        const decodedPathname = decodeURIComponent(url.pathname)
        const normalizedDecodedPathname = decodedPathname
          .replace(/\s+/g, '')
          .replace(/appendix_m+ath/gi, canonicalSegments[0])
          .replace(/linea+r-algebra-basics/gi, canonicalSegments[1])

        if (normalizedDecodedPathname !== decodedPathname) {
          const normalizedPathname = normalizedDecodedPathname
            .split('/')
            .map((segment) => encodeURIComponent(segment))
            .join('/')
          const redirectTarget = `${normalizedPathname}${url.search}`
          res.statusCode = 302
          res.setHeader('Location', redirectTarget)
          res.end()
          return
        }

        next()
      })
    }
  }
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function slugifySearchHeading(value) {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036F]/g, '')
    .replace(/[\x00-\x1f]/g, '')
    .replace(/[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function stripMarkdown(value) {
  return value
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_~>#-]/g, '')
    .trim()
}

function renderSearchMarkdown(src) {
  const html = []
  const slugCounts = new Map()
  let inFence = false

  for (const rawLine of src.replace(/^---[\s\S]*?---\n/, '').split('\n')) {
    const line = rawLine.trim()

    if (line.startsWith('```')) {
      inFence = !inFence
      continue
    }

    if (!line || inFence || line.startsWith(':::')) {
      continue
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const title = stripMarkdown(heading[2])
      const escapedTitle = escapeHtml(title)
      const baseSlug = slugifySearchHeading(title)
      const count = slugCounts.get(baseSlug) || 0
      slugCounts.set(baseSlug, count + 1)
      const slug = count === 0 ? baseSlug : `${baseSlug}-${count}`
      html.push(
        `<h${level}>${escapedTitle}<a class="header-anchor" href="#${slug}"></a></h${level}>`
      )
      continue
    }

    html.push(`<p>${escapeHtml(stripMarkdown(line))}</p>`)
  }

  return html.join('\n')
}

function isValidMathDelimiter(state, pos) {
  const max = state.posMax
  const prevChar = pos > 0 ? state.src.charCodeAt(pos - 1) : -1
  const nextChar = pos + 1 < max ? state.src.charCodeAt(pos + 1) : -1

  return {
    canOpen: nextChar !== 0x20 && nextChar !== 0x09,
    canClose:
      prevChar !== 0x20 &&
      prevChar !== 0x09 &&
      (nextChar < 0x30 || nextChar > 0x39)
  }
}

function mathInline(state, silent) {
  if (state.src[state.pos] !== '$') return false

  // Display math inline: $$...$$ inside a paragraph/list/blockquote line.
  // Must be tried before the single-$ branch so the closing $$ isn't read as
  // two consecutive empty inline-math delimiters.
  if (
    state.pos + 1 < state.posMax &&
    state.src[state.pos + 1] === '$'
  ) {
    const start = state.pos + 2
    const close = state.src.indexOf('$$', start)
    if (close !== -1 && close < state.posMax) {
      if (!silent) {
        const token = state.push('math_inline', 'math', 0)
        token.markup = '$$'
        token.content = state.src.slice(start, close)
        token.displayMode = true
      }
      state.pos = close + 2
      return true
    }
  }

  let delimiter = isValidMathDelimiter(state, state.pos)
  if (!delimiter.canOpen) {
    if (!silent) state.pending += '$'
    state.pos += 1
    return true
  }

  const start = state.pos + 1
  const max = state.posMax
  let match = start
  while ((match = state.src.indexOf('$', match)) !== -1) {
    if (match >= max) {
      match = -1
      break
    }
    let pos = match - 1
    while (state.src[pos] === '\\') pos -= 1
    if ((match - pos) % 2 === 1) break
    match += 1
  }

  if (match === -1) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (match - start === 0) {
    if (!silent) state.pending += '$$'
    state.pos = start + 1
    return true
  }

  delimiter = isValidMathDelimiter(state, match)
  if (!delimiter.canClose) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (!silent) {
    const token = state.push('math_inline', 'math', 0)
    token.markup = '$'
    token.content = state.src.slice(start, match)
  }

  state.pos = match + 1
  return true
}

function mathBlock(state, start, end, silent) {
  let pos = state.bMarks[start] + state.tShift[start]
  const max = state.eMarks[start]

  if (pos + 2 > max) return false
  if (state.src.slice(pos, pos + 2) !== '$$') return false

  pos += 2
  let firstLine = state.src.slice(pos, max)
  let lastLine = ''
  let found = false
  let next = start

  if (silent) return true
  if (firstLine.trim().slice(-2) === '$$') {
    firstLine = firstLine.trim().slice(0, -2)
    found = true
  }

  while (!found) {
    next++
    if (next >= end) break

    pos = state.bMarks[next] + state.tShift[next]
    const lineMax = state.eMarks[next]
    if (pos < lineMax && state.tShift[next] < state.blkIndent) break

    if (state.src.slice(pos, lineMax).trim().slice(-2) === '$$') {
      const lastPos = state.src.slice(0, lineMax).lastIndexOf('$$')
      lastLine = state.src.slice(pos, lastPos)
      found = true
    }
  }

  state.line = next + 1
  const token = state.push('math_block', 'math', 0)
  token.block = true
  token.content =
    (firstLine && firstLine.trim() ? `${firstLine}\n` : '') +
    state.getLines(start + 1, next, state.tShift[start], true) +
    (lastLine && lastLine.trim() ? lastLine : '')
  token.map = [start, state.line]
  token.markup = '$$'
  return true
}

function renderKatex(content, displayMode) {
  try {
    return katex.renderToString(content, {
      displayMode,
      output: 'htmlAndMathml',
      throwOnError: true,  // Enable error throwing for debugging
      strict: false,
      trust: true
    })
  } catch (error) {
    // Log detailed error information with file context
    const fs = require('fs');
    const path = require('path');
    const markdownFile = path.join(process.cwd(), 'docs/chapter07_ppo/ppo-math.md');

    console.error('\n' + '='.repeat(80))
    console.error('❌ KaTeX Rendering Error')
    console.error('='.repeat(80))
    console.error(`Mode: ${displayMode ? 'Display Math ($$...$$)' : 'Inline Math ($...$)'}`)
    console.error(`Expression: ${content.substring(0, 200)}${content.length > 200 ? '...' : ''}`)
    console.error(`Error: ${error.message}`)
    if (error.position !== undefined) {
      console.error(`Position in expression: ${error.position}`)
      const start = Math.max(0, error.position - 50)
      const end = Math.min(content.length, error.position + 50)
      console.error(`Context: ...${content.substring(start, end)}...`)
    }

    // Try to find this expression in the markdown file
    try {
      const mdContent = fs.readFileSync(markdownFile, 'utf8')
      const searchStr = content.substring(0, Math.min(100, content.length))
      const index = mdContent.indexOf(searchStr)
      if (index !== -1) {
        const lineNum = mdContent.substring(0, index).split('\n').length
        console.error(`Location: ${markdownFile}, approximately line ${lineNum}`)

        // Show surrounding lines
        const lines = mdContent.split('\n')
        const startLine = Math.max(0, lineNum - 3)
        const endLine = Math.min(lines.length, lineNum + 2)
        console.error('\nSurrounding context:')
        for (let i = startLine; i < endLine; i++) {
          const marker = (i + 1) === lineNum ? '>>> ' : '    '
          console.error(`${marker}${i + 1}: ${lines[i].substring(0, 100)}`)
        }
      }
    } catch (e) {
      // Ignore file reading errors
    }

    console.error('='.repeat(80) + '\n')

    // Return error HTML for visibility in the page
    return `<div style="background: #fee; border: 2px solid #c00; padding: 10px; margin: 10px 0; border-radius: 4px;">
      <strong style="color: #c00;">KaTeX Rendering Error</strong>
      <p style="margin: 5px 0;"><strong>Mode:</strong> ${displayMode ? 'Display' : 'Inline'}</p>
      <p style="margin: 5px 0;"><strong>Expression:</strong> <code>${content.substring(0, 150)}${content.length > 150 ? '...' : ''}</code></p>
      <p style="margin: 5px 0;"><strong>Error:</strong> ${error.message}</p>
      <p style="margin: 5px 0; font-size: 0.9em; color: #666;">Check browser console for detailed stack trace and file location.</p>
    </div>`
  }
}

function rescueMathInInline(md) {
  // Phase 1 (core, before inline): protect $...$ from emphasis/strong rules.
  // Replace inline-math in raw token content with unique placeholders so that
  // markdown's emphasis rule never sees the underscores inside formulas.
  let mathCounter = 0
  const mathStore = new Map()

  md.core.ruler.before('inline', 'math_inline_protect', function (state) {
    for (const token of state.tokens) {
      if (token.type !== 'inline') continue
      if (!token.content.includes('$')) continue

      token.content = token.content.replace(/\$([^\$]+)\$/g, function (match, formula) {
        const key = '\x01MATH' + (mathCounter++) + '\x01'
        mathStore.set(key, formula)
        return key
      })
    }
  })

  // Phase 2 (core, after inline): restore placeholders as math_inline tokens.
  md.core.ruler.after('inline', 'math_inline_restore', function (state) {
    for (const token of state.tokens) {
      if (token.type !== 'inline' || !token.children) continue

      let hasPlaceholder = false
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('\x01')) {
          hasPlaceholder = true
          break
        }
      }
      if (!hasPlaceholder) continue

      const newChildren = []
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('\x01')) {
          const parts = child.content.split(/(\x01MATH\d+\x01)/)
          for (const part of parts) {
            if (!part) continue
            if (part.startsWith('\x01MATH') && mathStore.has(part)) {
              const t = new state.Token('math_inline', 'math', 0)
              t.content = mathStore.get(part)
              t.markup = '$'
              newChildren.push(t)
            } else if (part.trim()) {
              const t = new state.Token('text', '', 0)
              t.content = part
              newChildren.push(t)
            }
          }
        } else {
          newChildren.push(child)
        }
      }
      token.children = newChildren
    }
  })
}

function katexMarkdown(md) {
  md.inline.ruler.push('math_inline', mathInline)
  md.block.ruler.after('blockquote', 'math_block', mathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  md.renderer.rules.math_inline = (tokens, idx) =>
    renderKatex(tokens[idx].content, tokens[idx].displayMode || false)
  md.renderer.rules.math_block = (tokens, idx) =>
    `<p>${renderKatex(tokens[idx].content, true)}</p>\n`
}

function footnoteTitlePlugin(md) {
  md.renderer.rules.footnote_block_open = (tokens, idx, options, env) => {
    const previousContentToken = [...tokens]
      .slice(0, idx)
      .reverse()
      .find((token) => token.type === 'inline' && token.content?.trim())
    const previousContent = (previousContentToken?.content || '')
      .replace(/[*_`#]/g, '')
      .trim()
    const hasManualTitle = /^(|References)[:：]?$/.test(
      previousContent
    )
    const title = 'References'
    const heading = hasManualTitle
      ? ''
      : `<div class="footnotes-title">${title}</div>\n`

    const separator = options.xhtmlOut
      ? '<hr class="footnotes-sep" />\n'
      : '<hr class="footnotes-sep">\n'

    return `${separator}<section class="footnotes">\n${heading}<ol class="footnotes-list">\n`
  }
}

function safeHeadingAttrs(md) {
  md.core.ruler.before('linkify', 'safe_heading_attrs', (state) => {
    for (let idx = 0; idx < state.tokens.length - 1; idx += 1) {
      const headingOpen = state.tokens[idx]
      const inline = state.tokens[idx + 1]

      if (headingOpen.type !== 'heading_open' || inline.type !== 'inline') {
        continue
      }

      const children = inline.children || []
      const lastText = [...children]
        .reverse()
        .find((token) => token.type === 'text')
      if (!lastText) continue

      const match = lastText.content.match(
        /\s*\{((?:[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)(?:\s+[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)*)\}$/
      )
      if (!match) continue

      const attrs = match[1].trim().split(/\s+/)
      const classes = []

      for (const attr of attrs) {
        if (attr.startsWith('#')) {
          headingOpen.attrSet('id', attr.slice(1))
        } else if (attr.startsWith('.')) {
          classes.push(attr.slice(1))
        }
      }

      if (classes.length) {
        headingOpen.attrJoin('class', classes.join(' '))
      }

      lastText.content = lastText.content.slice(0, match.index)
      inline.content = inline.content.replace(match[0], '')
    }
  })
}

function isExternalAsset(value) {
  return (
    /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(value) || value.startsWith('data:')
  )
}

function resolveMarkdownAsset(relativePagePath, src) {
  if (!src || isExternalAsset(src) || src.startsWith('/')) return null

  const hashIndex = src.indexOf('#')
  const queryIndex = src.indexOf('?')
  const suffixIndexCandidates = [hashIndex, queryIndex].filter(
    (idx) => idx >= 0
  )
  const suffixIndex = suffixIndexCandidates.length
    ? Math.min(...suffixIndexCandidates)
    : -1
  const cleanSrc = suffixIndex >= 0 ? src.slice(0, suffixIndex) : src
  const suffix = suffixIndex >= 0 ? src.slice(suffixIndex) : ''
  const pageDir = path.posix.dirname(relativePagePath || '')
  const sourcePath = path.posix
    .normalize(path.posix.join(pageDir, cleanSrc))
    .replace(/^\.\//, '')

  return {
    sourcePath,
    suffix
  }
}

function optimizedImagesPlugin(md) {
  const imageRule = md.renderer.rules.image

  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const src = token.attrGet('src')
    const resolved = resolveMarkdownAsset(env.relativePath, src)
    const asset = resolved && assetManifest.assets?.[resolved.sourcePath]

    if (asset?.status === 'optimized' && asset.optimized) {
      token.attrSet('src', `${asset.optimized}${resolved.suffix}`)
      token.attrSet('data-source-src', `/src/${resolved.sourcePath}`)
    }

    if (!token.attrGet('loading')) {
      token.attrSet('loading', 'lazy')
    }

    if (!token.attrGet('decoding')) {
      token.attrSet('decoding', 'async')
    }

    return imageRule(tokens, idx, options, env, self)
  }
}

function mermaidBlockIndex(tokens, idx) {
  let index = 0

  for (let i = 0; i <= idx; i += 1) {
    const info = tokens[i].info?.trim()
    if (info === 'mermaid' || info === 'mmd') {
      index += 1
    }
  }

  return index
}

function buildMermaidManifestMap() {
  return new Map(
    (assetManifest.mermaid || [])
      .filter((block) => block.status === 'optimized' && block.optimized)
      .map((block) => [`${block.page}:${block.index}`, block])
  )
}

const mermaidManifest = buildMermaidManifestMap()

function optimizedMermaidPlugin(md) {
  const fenceRule = md.renderer.rules.fence

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const info = token.info.trim()

    if (info === 'mermaid' || info === 'mmd') {
      const index = mermaidBlockIndex(tokens, idx)
      const block = mermaidManifest.get(`${env.relativePath}:${index}`)

      if (block?.optimized) {
        const src = md.utils.escapeHtml(block.optimized)
        const source = md.utils.escapeHtml(block.source)
        const page = md.utils.escapeHtml(block.page)

        return `<p class="mermaid-static"><img src="${src}" alt="Mermaid diagram" data-mermaid-viewer="true" data-source-src="/src/${source}" data-source-page="/src/${page}" data-source-index="${block.index}" loading="lazy" decoding="async"></p>\n`
      }
    }

    return fenceRule(tokens, idx, options, env, self)
  }
}

const zhNav = [
  { text: '', link: '/preface/intro' },
  { text: '', link: '/chapter01_cartpole/intro' },
  { text: '', link: '/chapter03_mdp/intro' },
  { text: '', link: '/chapter08_rlhf/intro' },
  {
    text: '',
    link: '/chapter11_vlm_rl/intro'
  }
]

const enNav = [
  { text: 'Preface', link: '/en/preface/intro' },
  { text: 'Fundamentals', link: '/en/chapter01_cartpole/intro' },
  { text: 'Core Theory', link: '/en/chapter03_mdp/intro' },
  { text: 'LLM RL', link: '/en/chapter08_rlhf/intro' },
  {
    text: 'Frontier Topics',
    link: '/en/chapter11_vlm_rl/intro'
  }
]

const zhSidebar = {
  '/': [
    {
      text: '',
      items: [
        { text: '', link: '/preface/intro' },
        { text: '', link: '/preface/brief-history/' },
        { text: '', link: '/preface/env-setup' }
      ]
    },
    {
      text: '',
      items: [
        {
          text: '1. CartPole ',
          link: '/chapter01_cartpole/intro',
          collapsed: false,
          items: [
            {
              text: '1.1 ',
              link: '/chapter01_cartpole/principles'
            },
            {
              text: '1.2 ',
              link: '/chapter01_cartpole/metrics'
            }
          ]
        },
        {
          text: '2. DPO ',
          link: '/chapter02_dpo/intro',
          collapsed: false,
          items: [
            {
              text: '2.1 DPO ',
              link: '/chapter02_dpo/principles'
            },
            {
              text: '2.2 ',
              link: '/chapter02_dpo/metrics'
            }
          ]
        },
        { text: '', link: '/summaries/part1-summary' }
      ]
    },
    {
      text: '',
      items: [
        {
          text: '3. MDP ',
          link: '/chapter03_mdp/intro',
          collapsed: false,
          items: [
            {
              text: '3.1 ',
              link: '/chapter03_mdp/bandit'
            },
            { text: '3.2 ', link: '/chapter03_mdp/mdp' },
            {
              text: '3.3 ',
              link: '/chapter03_mdp/value-bellman'
            },
            {
              text: '3.4 DP、MC  TD',
              link: '/chapter03_mdp/dp-mc-td'
            },
            { text: '3.5  Q  Q-Learning', link: '/chapter03_mdp/value-q' },
            {
              text: '3.6 ',
              link: '/chapter03_mdp/policy-objective'
            },
            {
              text: '3.7 ',
              link: '/chapter03_mdp/algorithm-taxonomy'
            },
            {
              text: '3.8 ',
              link: '/chapter03_mdp/reward-design'
            },
            {
              text: '3.9 ',
              link: '/chapter03_mdp/panorama'
            }
          ]
        },
        {
          text: '4.  Q ',
          link: '/chapter04_dqn/intro',
          collapsed: false,
          items: [
            {
              text: '4.1 DQN ',
              link: '/chapter04_dqn/from-q-to-dqn'
            },
            {
              text: '4.2 DQN ',
              link: '/chapter04_dqn/dqn-components'
            },
            {
              text: '4.3 ：LunarLander ',
              link: '/chapter04_dqn/lunar-lander'
            },
            {
              text: '4.4 DQN ',
              link: '/chapter04_dqn/dqn-family'
            },
            {
              text: '4.5 ：',
              link: '/chapter04_dqn/visual-game-projects'
            }
          ]
        },
        {
          text: '5. Policy-Based ',
          link: '/chapter05_policy_gradient/intro',
          collapsed: false,
          items: [
            {
              text: '5.1 ',
              link: '/chapter05_policy_gradient/pg-necessity'
            },
            {
              text: '5.2  REINFORCE',
              link: '/chapter05_policy_gradient/reinforce'
            },
            {
              text: '5.3 ： CartPole',
              link: '/chapter05_policy_gradient/cartpole'
            },
            {
              text: '5.4 ',
              link: '/chapter05_policy_gradient/pg-improvements'
            },
            {
              text: '5.5 ：',
              link: '/chapter05_policy_gradient/cartpole-baseline'
            }
          ]
        },
        {
          text: '6. Actor-Critic',
          link: '/chapter06_actor_critic/intro',
          collapsed: false,
          items: [
            {
              text: '6.1 ',
              link: '/chapter06_actor_critic/advantage-function'
            },
            {
              text: '6.2 Critic ',
              link: '/chapter06_actor_critic/critic-training'
            },
            {
              text: '6.3 Actor-Critic ',
              link: '/chapter06_actor_critic/actor-critic'
            },
            {
              text: '6.4 ：Pendulum ',
              link: '/chapter06_actor_critic/pendulum'
            },
            {
              text: '6.5 ：BipedalWalker ',
              link: '/chapter06_actor_critic/bipedalwalker'
            },
            {
              text: '6.6 Actor-Critic ',
              link: '/chapter06_actor_critic/ac-frontier'
            }
          ]
        },
        {
          text: '7. PPO',
          link: '/chapter07_ppo/intro',
          collapsed: false,
          items: [
            {
              text: '7.1 ：BipedalWalker ',
              link: '/chapter07_ppo/ppo-bipedal-walker'
            },
            {
              text: '7.2 PPO ',
              link: '/chapter07_ppo/ppo-math'
            },
            {
              text: '7.3 ',
              link: '/chapter07_ppo/trust-region-clipping'
            },
            {
              text: '7.4 ',
              link: '/chapter07_ppo/gae-reward-model'
            },
            {
              text: '7.5 PPO ',
              link: '/chapter07_ppo/ppo-game-benchmark'
            },
            {
              text: '7.6  RL',
              link: '/chapter07_ppo/rl-long-horizon-planning'
            }
          ]
        },
        { text: '', link: '/summaries/part2-summary' }
      ]
    },
    {
      text: ' RL',
      items: [
        {
          text: '8. RLHF ',
          link: '/chapter08_rlhf/intro',
          collapsed: false,
          items: [
            {
              text: '8.1 Base ',
              link: '/chapter08_rlhf/base-model-to-assistant'
            },
            {
              text: '8.2 RLHF ',
              link: '/chapter08_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '8.3 SFT ',
              link: '/chapter08_rlhf/imitation-learning-pipeline'
            },
            {
              text: '8.4 ',
              link: '/chapter08_rlhf/reward-function-design'
            },
            {
              text: '8.5 PPO-RLHF ',
              link: '/chapter08_rlhf/ppo-rlhf-loop'
            },
            {
              text: '8.6 ',
              link: '/chapter08_rlhf/evaluation'
            },
            {
              text: '8.7 ：veRL PPO  GSM8K',
              link: '/chapter08_rlhf/verl-ppo-gsm8k'
            },
            {
              text: '8.8 ',
              link: '/chapter08_rlhf/extended-practice'
            }
          ]
        },
        {
          text: '9. ',
          link: '/chapter09_alignment/intro',
          collapsed: false,
          items: [
            {
              text: '9.1 DPO 、',
              link: '/chapter09_alignment/dpo-theory-and-family'
            },
            {
              text: '9.2 GRPO ',
              link: '/chapter09_grpo_rlvr/grpo-practice-and-mechanism'
            },
            {
              text: '9.3 R1-Zero ',
              link: '/chapter09_grpo_rlvr/deepseek-dapo'
            },
            {
              text: '9.4 RLVR ',
              link: '/chapter09_grpo_rlvr/rlvr'
            },
            {
              text: '9.5 ： API  GRPO',
              link: '/chapter09_grpo_rlvr/financial-tool-calling-grpo'
            },
            {
              text: '9.6 OPD ',
              link: '/chapter09_grpo_rlvr/on-policy-distillation'
            },
            {
              text: '9.7 ： veRL ',
              link: '/chapter09_grpo_rlvr/verl-code-sandbox'
            },
            {
              text: '9.8 ',
              link: '/chapter09_alignment/industrial-post-training'
            }
          ]
        },
        {
          text: '10. Agentic RL',
          link: '/chapter10_agentic_rl/intro',
          collapsed: false,
          items: [
            {
              text: '10.1 ',
              link: '/chapter10_agentic_rl/multi-turn-rl'
            },
            {
              text: '10.2 ',
              link: '/chapter10_agentic_rl/tool-use-and-trajectory'
            },
            {
              text: '10.3 ',
              link: '/chapter10_agentic_rl/industrial-evaluation'
            },
            {
              text: '10.4 ：Agent ',
              link: '/chapter10_agentic_rl/agent-data-swe-smith'
            },
            {
              text: '10.5 ：rLLM DeepCoder',
              link: '/chapter10_agentic_rl/rllm-deepcoder-lab'
            },
            {
              text: '10.6 ： Agent',
              link: '/chapter10_agentic_rl/rllm-finqa-lab'
            },
            {
              text: '10.7 Deep Research',
              link: '/chapter10_agentic_rl/deep-research-agent'
            },
            {
              text: '10.8 Agentic RL ',
              link: '/chapter10_agentic_rl/build-agentic-training-system'
            },
            {
              text: '10.9 ',
              link: '/chapter10_agentic_rl/extended-readings'
            }
          ]
        },
        { text: '', link: '/summaries/part3-summary' }
      ]
    },
    {
      text: '',
      items: [
        {
          text: '11. VLM ',
          link: '/chapter11_vlm_rl/intro',
          collapsed: false,
          items: [
            {
              text: '11.1 VLM ',
              link: '/chapter11_vlm_rl/vlm-grpo-hands-on'
            },
            {
              text: '11.2 ',
              link: '/chapter11_vlm_rl/vlm-challenges'
            },
            {
              text: '11.3 VLM RL ',
              link: '/chapter11_vlm_rl/vlm-frameworks'
            },
            {
              text: '11.4  RL',
              link: '/chapter11_vlm_rl/visual-generation-rl'
            },
            {
              text: '11.5 ：GeoQA ',
              link: '/chapter11_vlm_rl/easyr1-geoqa'
            }
          ]
        },
        {
          text: '12. ',
          link: '/chapter12_future_trends/intro',
          collapsed: false,
          items: [
            {
              text: '12.1 ',
              link: '/chapter12_future_trends/embodied-intelligence/'
            },
            {
              text: '12.2 ',
              link: '/chapter12_future_trends/embodied-intelligence/model-based-rl/'
            },
            {
              text: '12.3 ',
              link: '/chapter12_future_trends/self-play-outlook/'
            },
            {
              text: '12.4 ',
              link: '/chapter12_future_trends/llm-multi-agent-rl/'
            },
            {
              text: '12.5 ',
              link: '/chapter12_future_trends/offline-rl/'
            },
            {
              text: '12.6 ',
              link: '/chapter12_future_trends/rl-scaling-outlook'
            }
          ]
        },
        { text: '', link: '/summaries/part4-summary' }
      ]
    },
    {
      text: '',
      items: [
        {
          text: 'A. ',
          link: '/appendix_common_pitfalls/intro'
        },
        {
          text: 'B. RL ',
          link: '/appendix_industrial_training/intro',
          collapsed: false,
          items: [
            {
              text: 'B.1 ',
              link: '/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'B.2 Agent ',
              link: '/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'B.3 ',
              link: '/appendix_industrial_training/evaluation-badcase'
            },
            {
              text: 'B.4 ',
              link: '/appendix_industrial_training/metrics-glossary'
            },
            {
              text: 'B.5 ',
              link: '/appendix_industrial_training/industrial-exercises'
            }
          ]
        },
        {
          text: 'C. ',
          link: '/appendix_code_cheatsheet/intro',
          collapsed: false,
          items: [
            {
              text: 'C.1 SFT  KL',
              link: '/appendix_code_cheatsheet/sft-kl'
            },
            {
              text: 'C.2 PPO  GAE',
              link: '/appendix_code_cheatsheet/ppo-gae'
            },
            {
              text: 'C.3 DPO ',
              link: '/appendix_code_cheatsheet/dpo-family'
            },
            {
              text: 'C.4 GRPO ',
              link: '/appendix_code_cheatsheet/grpo-rlvr'
            },
            {
              text: 'C.5 Softmax ',
              link: '/appendix_code_cheatsheet/softmax-ce'
            },
            {
              text: 'C.6 ',
              link: '/appendix_code_cheatsheet/top-k-top-p'
            },
            {
              text: 'C.7 ',
              link: '/appendix_code_cheatsheet/attention-mha'
            },
            {
              text: 'C.8 DAPO',
              link: '/appendix_code_cheatsheet/dapo'
            }
          ]
        },
        {
          text: 'D. ',
          link: '/appendix_game_projects/intro'
        },
        {
          text: 'E. ',
          link: '/appendix_math/intro',
          collapsed: false,
          items: [
            {
              text: 'E.1 ',
              link: '/appendix_math/linear-algebra',
              collapsed: true,
              items: [
                {
                  text: '',
                  link: '/appendix_math/linear-algebra-basics'
                },
                {
                  text: '',
                  link: '/appendix_math/linear-algebra-bellman'
                },
                {
                  text: '',
                  link: '/appendix_math/linear-algebra-function-approx'
                },
                {
                  text: '',
                  link: '/appendix_math/linear-algebra-advanced'
                },
                {
                  text: '',
                  link: '/appendix_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.2 、',
              link: '/appendix_math/probability-statistics',
              collapsed: true,
              items: [
                { text: '', link: '/appendix_math/probability-basics' },
                {
                  text: '',
                  link: '/appendix_math/probability-value'
                },
                {
                  text: '',
                  link: '/appendix_math/probability-sampling'
                },
                {
                  text: ' GAE',
                  link: '/appendix_math/probability-trajectory-td'
                },
                {
                  text: '',
                  link: '/appendix_math/probability-bellman-advanced'
                },
                {
                  text: '',
                  link: '/appendix_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.3 ',
              link: '/appendix_math/calculus-optimization',
              collapsed: true,
              items: [
                { text: '', link: '/appendix_math/calculus-basics' },
                {
                  text: '',
                  link: '/appendix_math/calculus-policy-gradient'
                },
                { text: 'PPO  Adam', link: '/appendix_math/calculus-ppo' },
                {
                  text: '',
                  link: '/appendix_math/calculus-derivations'
                },
                {
                  text: '',
                  link: '/appendix_math/calculus-advanced-formulas'
                },
                {
                  text: '',
                  link: '/appendix_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.4 ',
              link: '/appendix_math/information-theory',
              collapsed: true,
              items: [
                { text: '', link: '/appendix_math/information-basics' },
                {
                  text: ' KL',
                  link: '/appendix_math/information-cross-entropy-kl'
                },
                {
                  text: 'RLHF  DPO',
                  link: '/appendix_math/information-rlhf-dpo'
                },
                {
                  text: '',
                  link: '/appendix_math/information-mutual-info'
                },
                {
                  text: '',
                  link: '/appendix_math/information-advanced-formulas'
                },
                {
                  text: '',
                  link: '/appendix_math/information-formulas-exercises'
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

const enSidebar = {
  '/en/': [
    {
      text: 'Preface',
      items: [
        { text: 'Course Overview', link: '/en/preface/intro' },
        { text: 'A Brief History of RL', link: '/en/preface/brief-history' },
        { text: 'Environment Setup', link: '/en/preface/env-setup' }
      ]
    },
    {
      text: 'Fundamentals',
      items: [
        {
          text: '1. CartPole Balancing',
          link: '/en/chapter01_cartpole/intro',
          collapsed: false,
          items: [
            {
              text: '1.1 Core Concepts',
              link: '/en/chapter01_cartpole/principles'
            },
            {
              text: '1.2 Training Metrics',
              link: '/en/chapter01_cartpole/metrics'
            }
          ]
        },
        {
          text: '2. DPO Preference Tuning',
          link: '/en/chapter02_dpo/intro',
          collapsed: false,
          items: [
            {
              text: '2.1 DPO Derivation',
              link: '/en/chapter02_dpo/principles'
            },
            { text: '2.2 Training Metrics', link: '/en/chapter02_dpo/metrics' }
          ]
        },
        { text: 'Part I Summary', link: '/en/summaries/part1-summary' }
      ]
    },
    {
      text: 'Core Theory and Methods',
      items: [
        {
          text: '3. MDP and Value Functions',
          link: '/en/chapter03_mdp/intro',
          collapsed: false,
          items: [
            {
              text: '3.1 The Two-Armed Bandit',
              link: '/en/chapter03_mdp/bandit'
            },
            {
              text: '3.2 Markov Decision Processes',
              link: '/en/chapter03_mdp/mdp'
            },
            {
              text: '3.3 Value Functions & Bellman',
              link: '/en/chapter03_mdp/value-bellman'
            },
            { text: '3.4 DP, MC, and TD', link: '/en/chapter03_mdp/dp-mc-td' },
            {
              text: '3.5 From V to Q-Learning',
              link: '/en/chapter03_mdp/value-q'
            },
            {
              text: '3.6 From Value to Policy',
              link: '/en/chapter03_mdp/policy-objective'
            },
            {
              text: '3.7 Where Does Data Come From',
              link: '/en/chapter03_mdp/algorithm-taxonomy'
            },
            {
              text: '3.8 Reward Function Design',
              link: '/en/chapter03_mdp/reward-design'
            },
            { text: '3.9 Chapter Summary', link: '/en/chapter03_mdp/panorama' }
          ]
        },
        {
          text: '4. Deep Q-Networks',
          link: '/en/chapter04_dqn/intro',
          collapsed: false,
          items: [
            {
              text: '4.1 Why DQN Is Needed',
              link: '/en/chapter04_dqn/from-q-to-dqn'
            },
            {
              text: '4.2 DQN Architecture',
              link: '/en/chapter04_dqn/dqn-components'
            },
            {
              text: '4.3 Hands-On: LunarLander',
              link: '/en/chapter04_dqn/lunar-lander'
            },
            {
              text: '4.4 DQN Improvement Family',
              link: '/en/chapter04_dqn/dqn-family'
            },
            {
              text: '4.5 Hands-On: Visual Games',
              link: '/en/chapter04_dqn/visual-game-projects'
            }
          ]
        },
        {
          text: '5. Policy-Based Methods',
          link: '/en/chapter05_policy_gradient/intro',
          collapsed: false,
          items: [
            {
              text: '5.1 Why Policy Gradients',
              link: '/en/chapter05_policy_gradient/pg-necessity'
            },
            {
              text: '5.2 Policy Gradient & REINFORCE',
              link: '/en/chapter05_policy_gradient/reinforce'
            },
            {
              text: '5.3 Hands-On: PG CartPole',
              link: '/en/chapter05_policy_gradient/cartpole'
            },
            {
              text: '5.4 Variance and Baselines',
              link: '/en/chapter05_policy_gradient/pg-improvements'
            },
            {
              text: '5.5 Hands-On: PG with Baseline',
              link: '/en/chapter05_policy_gradient/cartpole-baseline'
            }
          ]
        },
        {
          text: '6. Actor-Critic',
          link: '/en/chapter06_actor_critic/intro',
          collapsed: false,
          items: [
            {
              text: '6.1 The Advantage Function',
              link: '/en/chapter06_actor_critic/advantage-function'
            },
            {
              text: '6.2 Training the Critic',
              link: '/en/chapter06_actor_critic/critic-training'
            },
            {
              text: '6.3 Actor-Critic Architecture',
              link: '/en/chapter06_actor_critic/actor-critic'
            },
            {
              text: '6.4 Hands-On: Pendulum',
              link: '/en/chapter06_actor_critic/pendulum'
            },
            {
              text: '6.5 Hands-On: BipedalWalker',
              link: '/en/chapter06_actor_critic/bipedalwalker'
            },
            {
              text: '6.6 Actor-Critic at Scale',
              link: '/en/chapter06_actor_critic/ac-frontier'
            }
          ]
        },
        {
          text: '7. PPO',
          link: '/en/chapter07_ppo/intro',
          collapsed: false,
          items: [
            {
              text: '7.1 Hands-On: BipedalWalker',
              link: '/en/chapter07_ppo/ppo-bipedal-walker'
            },
            { text: '7.2 PPO Derivation', link: '/en/chapter07_ppo/ppo-math' },
            {
              text: '7.3 Constraint Mechanisms for Policy Updates',
              link: '/en/chapter07_ppo/trust-region-clipping'
            },
            {
              text: '7.4 Advantage Estimation and Reward Modeling',
              link: '/en/chapter07_ppo/gae-reward-model'
            },
            {
              text: '7.5 PPO Game Benchmarks',
              link: '/en/chapter07_ppo/ppo-game-benchmark'
            },
            {
              text: '7.6 RL in Long-Horizon Tasks',
              link: '/en/chapter07_ppo/rl-long-horizon-planning'
            }
          ]
        },
        { text: 'Part II Summary', link: '/en/summaries/part2-summary' }
      ]
    },
    {
      text: 'LLM Reinforcement Learning',
      items: [
        {
          text: '8. The RLHF Pipeline',
          link: '/en/chapter08_rlhf/intro',
          collapsed: false,
          items: [
            {
              text: '8.1 Base Model to Assistant',
              link: '/en/chapter08_rlhf/base-model-to-assistant'
            },
            {
              text: '8.2 RLHF Pipeline',
              link: '/en/chapter08_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '8.3 SFT Instruction Tuning',
              link: '/en/chapter08_rlhf/imitation-learning-pipeline'
            },
            {
              text: '8.4 Reward Models',
              link: '/en/chapter08_rlhf/reward-function-design'
            },
            {
              text: '8.5 PPO-RLHF Alignment',
              link: '/en/chapter08_rlhf/ppo-rlhf-loop'
            },
            {
              text: '8.6 Evaluation & Reward Hacking',
              link: '/en/chapter08_rlhf/evaluation'
            },
            {
              text: '8.7 Hands-on: veRL PPO on GSM8K',
              link: '/en/chapter08_rlhf/verl-ppo-gsm8k'
            },
            {
              text: '8.8 Extended Practice',
              link: '/en/chapter08_rlhf/extended-practice'
            }
          ]
        },
        {
          text: '9. Post-Training Alignment',
          link: '/en/chapter09_alignment/intro',
          collapsed: false,
          items: [
            {
              text: '9.1 DPO Theory and Selection',
              link: '/en/chapter09_alignment/dpo-theory-and-family'
            },
            {
              text: '9.2 GRPO Training',
              link: '/en/chapter09_grpo_rlvr/grpo-practice-and-mechanism'
            },
            {
              text: '9.3 The R1-Zero Paradigm',
              link: '/en/chapter09_grpo_rlvr/deepseek-dapo'
            },
            {
              text: '9.4 RLVR: Verifiable Rewards',
              link: '/en/chapter09_grpo_rlvr/rlvr'
            },
            {
              text: '9.5 On-Policy Distillation',
              link: '/en/chapter09_grpo_rlvr/on-policy-distillation'
            },
            {
              text: '9.7 Industrial Post-Training',
              link: '/en/chapter09_alignment/industrial-post-training'
            }
          ]
        },
        {
          text: '10. Agentic RL',
          link: '/en/chapter10_agentic_rl/intro',
          collapsed: false,
          items: [
            {
              text: '10.1 Multi-Turn Interaction',
              link: '/en/chapter10_agentic_rl/multi-turn-rl'
            },
            {
              text: '10.2 Tool Use',
              link: '/en/chapter10_agentic_rl/tool-use-and-trajectory'
            },
            {
              text: '10.3 Benchmarks & Cases',
              link: '/en/chapter10_agentic_rl/industrial-evaluation'
            },
            {
              text: '10.4 Hands-On: Agent Data',
              link: '/en/chapter10_agentic_rl/agent-data-swe-smith'
            },
            {
              text: '10.5 Hands-On: DeepCoder',
              link: '/en/chapter10_agentic_rl/rllm-deepcoder-lab'
            },
            {
              text: '10.6 Hands-On: FinQA Agent',
              link: '/en/chapter10_agentic_rl/rllm-finqa-lab'
            },
            {
              text: '10.7 Deep Research',
              link: '/en/chapter10_agentic_rl/deep-research-agent'
            },
            {
              text: '10.8 Agentic Training Systems',
              link: '/en/chapter10_agentic_rl/build-agentic-training-system'
            },
            {
              text: '10.9 Extended Readings',
              link: '/en/chapter10_agentic_rl/extended-readings'
            }
          ]
        },
        { text: 'Part III Summary', link: '/en/summaries/part3-summary' }
      ]
    },
    {
      text: 'Frontier Topics',
      items: [
        {
          text: '11. VLM Reinforcement Learning',
          link: '/en/chapter11_vlm_rl/intro',
          collapsed: false,
          items: [
            {
              text: '11.1 VLM RL Training',
              link: '/en/chapter11_vlm_rl/vlm-grpo-hands-on'
            },
            {
              text: '11.2 Visual Reward Signals',
              link: '/en/chapter11_vlm_rl/vlm-challenges'
            },
            {
              text: '11.3 VLM RL Frameworks',
              link: '/en/chapter11_vlm_rl/vlm-frameworks'
            },
            {
              text: '11.4 Visual Generation RL',
              link: '/en/chapter11_vlm_rl/visual-generation-rl'
            },
            {
              text: '11.5 Hands-On: GeoQA',
              link: '/en/chapter11_vlm_rl/easyr1-geoqa'
            }
          ]
        },
        {
          text: '12. Future Trends',
          link: '/en/chapter12_future_trends/intro',
          collapsed: false,
          items: [
            {
              text: '12.1 Embodied Intelligence',
              link: '/en/chapter12_future_trends/embodied-intelligence/'
            },
            {
              text: '12.2 Model-Based RL',
              link: '/en/chapter12_future_trends/embodied-intelligence/model-based-rl'
            },
            {
              text: '12.3 Self-Play',
              link: '/en/chapter12_future_trends/self-play-outlook/'
            },
            {
              text: '12.4 Multi-Agent RL',
              link: '/en/chapter12_future_trends/llm-multi-agent-rl/'
            },
            {
              text: '12.5 Offline RL',
              link: '/en/chapter12_future_trends/offline-rl/'
            },
            {
              text: '12.6 Scaling Trends',
              link: '/en/chapter12_future_trends/rl-scaling-outlook'
            }
          ]
        },
        { text: 'Part IV Summary', link: '/en/summaries/part4-summary' }
      ]
    },
    {
      text: 'Appendices',
      items: [
        {
          text: 'A. Training Debugging Guide',
          link: '/en/appendix_common_pitfalls/intro'
        },
        {
          text: 'B. RL Engineering Practice',
          link: '/en/appendix_industrial_training/intro',
          collapsed: false,
          items: [
            {
              text: 'B.1 Training Infrastructure',
              link: '/en/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'B.2 Agent Sandbox',
              link: '/en/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'B.3 Evaluation Benchmarks',
              link: '/en/appendix_industrial_training/evaluation-badcase'
            },
            {
              text: 'B.4 Metrics Glossary',
              link: '/en/appendix_industrial_training/metrics-glossary'
            },
            {
              text: 'B.5 Industrial Exercises',
              link: '/en/appendix_industrial_training/industrial-exercises'
            }
          ]
        },
        {
          text: 'C. Code Cheatsheet',
          link: '/en/appendix_code_cheatsheet/intro',
          collapsed: false,
          items: [
            {
              text: 'C.1 SFT and KL',
              link: '/en/appendix_code_cheatsheet/sft-kl'
            },
            {
              text: 'C.2 PPO and GAE',
              link: '/en/appendix_code_cheatsheet/ppo-gae'
            },
            {
              text: 'C.3 DPO Family',
              link: '/en/appendix_code_cheatsheet/dpo-family'
            },
            {
              text: 'C.4 GRPO and Reward Models',
              link: '/en/appendix_code_cheatsheet/grpo-rlvr'
            },
            {
              text: 'C.5 Softmax & Cross-Entropy',
              link: '/en/appendix_code_cheatsheet/softmax-ce'
            },
            {
              text: 'C.6 Sampling Methods',
              link: '/en/appendix_code_cheatsheet/top-k-top-p'
            },
            {
              text: 'C.7 Attention Mechanism',
              link: '/en/appendix_code_cheatsheet/attention-mha'
            },
            { text: 'C.8 DAPO', link: '/en/appendix_code_cheatsheet/dapo' }
          ]
        },
        {
          text: 'D. Learning Resources',
          link: '/en/appendix_game_projects/intro'
        },
        {
          text: 'E. Math Foundations for RL',
          link: '/en/appendix_math/intro',
          collapsed: false,
          items: [
            {
              text: 'E.1 Linear Algebra',
              link: '/en/appendix_math/linear-algebra',
              collapsed: true,
              items: [
                {
                  text: 'Basic Objects',
                  link: '/en/appendix_math/linear-algebra-basics'
                },
                {
                  text: 'Bellman Matrix',
                  link: '/en/appendix_math/linear-algebra-bellman'
                },
                {
                  text: 'Function Approximation',
                  link: '/en/appendix_math/linear-algebra-function-approx'
                },
                {
                  text: 'Convergence & Trust Regions',
                  link: '/en/appendix_math/linear-algebra-advanced'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.2 Probability & Estimation',
              link: '/en/appendix_math/probability-statistics',
              collapsed: true,
              items: [
                {
                  text: 'Probability Basics',
                  link: '/en/appendix_math/probability-basics'
                },
                {
                  text: 'Returns and Value',
                  link: '/en/appendix_math/probability-value'
                },
                {
                  text: 'Sampling & Estimation',
                  link: '/en/appendix_math/probability-sampling'
                },
                {
                  text: 'Trajectories and GAE',
                  link: '/en/appendix_math/probability-trajectory-td'
                },
                {
                  text: 'Bellman Expectations',
                  link: '/en/appendix_math/probability-bellman-advanced'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.3 Calculus & Optimization',
              link: '/en/appendix_math/calculus-optimization',
              collapsed: true,
              items: [
                {
                  text: 'Derivatives & Gradients',
                  link: '/en/appendix_math/calculus-basics'
                },
                {
                  text: 'Policy Gradient',
                  link: '/en/appendix_math/calculus-policy-gradient'
                },
                {
                  text: 'PPO and Adam',
                  link: '/en/appendix_math/calculus-ppo'
                },
                {
                  text: 'Derivation Tools',
                  link: '/en/appendix_math/calculus-derivations'
                },
                {
                  text: 'Complete Formulas',
                  link: '/en/appendix_math/calculus-advanced-formulas'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'E.4 Information Theory',
              link: '/en/appendix_math/information-theory',
              collapsed: true,
              items: [
                {
                  text: 'Entropy & Exploration',
                  link: '/en/appendix_math/information-basics'
                },
                {
                  text: 'Cross-Entropy & KL',
                  link: '/en/appendix_math/information-cross-entropy-kl'
                },
                {
                  text: 'RLHF and DPO',
                  link: '/en/appendix_math/information-rlhf-dpo'
                },
                {
                  text: 'Mutual Information',
                  link: '/en/appendix_math/information-mutual-info'
                },
                {
                  text: 'Complete Formulas',
                  link: '/en/appendix_math/information-advanced-formulas'
                },
                {
                  text: 'Formulas & Exercises',
                  link: '/en/appendix_math/information-formulas-exercises'
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

const logger = createLogger()
const originalWarn = logger.warn
logger.warn = (msg, options) => {
  if (msg.includes('Failed to resolve "/@siteData"')) return
  originalWarn(msg, options)
}

export default defineConfig({
  lang: 'en-US',
  title: 'Hands-on Modern RL',
  description:
    'Modern Reinforcement Learning in Practice — From Code to Theory',
  base,
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    image: {
      lazyLoading: true
    },
    attrs: {
      disable: true
    },
    config: (md) => {
      safeHeadingAttrs(md)
      optimizedImagesPlugin(md)
      md.use(markdownItFootnote)
      footnoteTitlePlugin(md)
      katexMarkdown(md)
      MermaidMarkdown(md)
      optimizedMermaidPlugin(md)
      // Custom "output" container for displaying code running results
      md.use(markdownItContainer, 'output', {
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            const title = tokens[idx].info.trim().slice(6).trim() || ''
            return `<div class="custom-block output"><p class="custom-block-title">${title}</p>\n`
          }
          return '</div>\n'
        }
      })
    }
  },
  vite: {
    customLogger: logger,
    plugins: [mermaidConfigPlugin(), normalizeBrokenDocPathPlugin(), katexDevPlugin()],
    optimizeDeps: {
      include: [
        '@braintree/sanitize-url',
        'cytoscape',
        'cytoscape-cose-bilkent',
        'dayjs',
        'debug'
      ]
    },
    resolve: {
      alias: {
        'dayjs/plugin/advancedFormat.js': 'dayjs/esm/plugin/advancedFormat',
        'dayjs/plugin/customParseFormat.js':
          'dayjs/esm/plugin/customParseFormat',
        'dayjs/plugin/isoWeek.js': 'dayjs/esm/plugin/isoWeek',
        'cytoscape/dist/cytoscape.umd.js': 'cytoscape/dist/cytoscape.esm.js'
      }
    }
  },
  ignoreDeadLinks: true,
  head: [
    ['link', { rel: 'icon', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#3f51b5' }],
    [
      'meta',
      { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }
    ],
    ['meta', { name: 'author', content: 'letslego' }],
    ['meta', { name: 'robots', content: 'index,follow' }],
    ['meta', { property: 'og:title', content: 'Hands-on Modern RL' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          '：、LLM 、RLVR '
      }
    ],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:url', content: siteUrl }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    // KaTeX client-side rendering for dev mode
    ['script', { src: `${base}katex.min.js` }],
    ['script', { src: `${base}auto-render.min.js` }]
  ],
  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      link: '/',
      title: 'Hands-on Modern RL',
      description: 'Modern Reinforcement Learning in Practice — From Code to Theory',
      themeConfig: {
        nav: enNav,
        sidebar: enSidebar,
        editLink: {
          pattern: editLinkPattern,
          text: 'Edit this page on GitHub'
        },
        footer: {
          message: 'Hands-on Modern RL course',
          copyright: 'Copyright © letslego'
        },
        outline: {
          level: [2, 3],
          label: 'Outline'
        },
        lastUpdated: {
          text: 'Last updated'
        },
        docFooter: {
          prev: 'Previous page',
          next: 'Next page'
        },
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        returnToTopLabel: 'Return to top',
        langMenuLabel: 'Change language',
        skipToContentLabel: 'Skip to content',
        notFound: {
          title: 'Page not found',
          quote: 'This page is missing. Try going back to the home page.',
          link: '/',
          linkText: 'Take me home',
          linkLabel: 'Go to home page'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'Hands-on Modern RL',
      description:
        'Modern Reinforcement Learning in Practice — From Code to Theory',
      themeConfig: {
        nav: enNav,
        sidebar: enSidebar,
        editLink: {
          pattern: editLinkPattern,
          text: 'Edit this page on GitHub'
        },
        footer: {
          message: 'Hands-on Modern RL course',
          copyright: 'Copyright © letslego'
        },
        outline: {
          level: [2, 3],
          label: 'Outline'
        },
        lastUpdated: {
          text: 'Last updated'
        },
        docFooter: {
          prev: 'Previous page',
          next: 'Next page'
        },
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        returnToTopLabel: 'Return to top',
        langMenuLabel: 'Change language',
        skipToContentLabel: 'Skip to content',
        notFound: {
          title: 'Page not found',
          quote:
            'This page is missing. Try jumping back in from the English home page.',
          link: '/en/',
          linkText: 'Take me to English home',
          linkLabel: 'Go to English home'
        }
      }
    }
  },
  themeConfig: {
    logo: '/readme/logo-symbol.svg',
    siteTitle: 'Hands on Modern RL',
    nav: enNav,
    sidebar: enSidebar,
    socialLinks: [
      { icon: 'github', link: `https://github.com/${owner}/${repo}` }
    ],
    search: enableLocalSearch
      ? {
          provider: 'local',
          options: {
            _render: renderSearchMarkdown
          }
        }
      : undefined,
    editLink: {
      pattern: editLinkPattern,
      text: 'Edit this page on GitHub'
    },
    footer: {
      message: 'Built for reusable bilingual course delivery',
      copyright: 'Copyright © letslego'
    },
    outline: {
      level: [2, 3],
      label: 'Outline'
    }
  }
})
