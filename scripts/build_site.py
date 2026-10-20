"""Gera index.html (home) e aplica o tema Quant Notebooks aos HTMLs exportados pelo marimo.

Uso: python scripts/build_site.py   (rode depois de `marimo export html`)
Idempotente: re-injeta o tema entre marcadores <!--qn-theme--> ... <!--/qn-theme-->.
"""
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "assets/notebooks.json").read_text(encoding="utf-8"))
REPO, PORTFOLIO = CFG["repo"], CFG["portfolio"]
VER = "3"  # cache-busting dos assets


def minutes(py: Path) -> int:
    src = py.read_text(encoding="utf-8") if py.exists() else ""
    md = " ".join(re.findall(r'mo\.md\(\s*r?f?"""(.*?)"""', src, re.S))
    words = len(md.split())
    code = max(0, len(src.splitlines()) - md.count("\n"))
    return max(3, round(words / 200 + code / 45))


GA_ID = "G-QCBFGR39WG"
GA_MARK = re.compile(r"<!--qn-ga-->.*?<!--/qn-ga-->", re.S)
ICON_RE = re.compile(r'<link rel="(?:icon|shortcut icon|apple-touch-icon)"[^>]*>\s*', re.I)


def head_extra(root: str, notebook: str) -> str:
    """Favicon + GA4 (idempotente entre <!--qn-ga--> e <!--/qn-ga-->)."""
    return (f'<!--qn-ga--><link rel="icon" href="{root}favicon.ico" sizes="48x48">'
            f'<link rel="icon" type="image/svg+xml" href="{root}favicon.svg">'
            f'<link rel="icon" type="image/png" sizes="32x32" href="{root}favicon-32.png">'
            f'<link rel="apple-touch-icon" href="{root}apple-touch-icon.png">'
            f'<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>'
            f"<script>window.dataLayer=window.dataLayer||[];function gtag(){{dataLayer.push(arguments)}}"
            f"gtag('js',new Date());gtag('set',{{content_group:'quant-notebooks',notebook:'{notebook}'}});"
            f"gtag('config','{GA_ID}');</script>"
            f'<script defer src="{root}assets/ga.js?v={VER}"></script><!--/qn-ga-->')


def esc(s):
    return html.escape(str(s), quote=True)


nbs = CFG["notebooks"]
for nb in nbs:
    nb["minutes"] = minutes(ROOT / (nb["path"] + ".py"))
    nb["exists"] = (ROOT / (nb["path"] + ".html")).exists()
published = [n for n in nbs if n["exists"]]

# ---------- notebooks ----------
MARK = re.compile(r"<!--qn-theme-->.*?<!--/qn-theme-->", re.S)
for nb in published:
    f = ROOT / (nb["path"] + ".html")
    doc = GA_MARK.sub("", MARK.sub("", f.read_text(encoding="utf-8")))
    doc = ICON_RE.sub("", doc)
    related = [{"path": o["path"], "title": o["title"]} for o in nbs
               if o is not nb and o["exists"] and (o["theme"] == nb["theme"] or set(o["tags"]) & set(nb["tags"]))][:3]
    if len(related) < 2:
        related += [{"path": o["path"], "title": o["title"]} for o in published if o is not nb and o["path"] not in [r["path"] for r in related]][: 2 - len(related)]
    meta = {
        "title": nb["title"], "short": nb.get("short", nb["title"].split(" (")[0] if len(nb["title"]) > 34 else nb["title"]),
        "theme": nb["theme"], "level": nb["level"], "date": nb["date"], "minutes": nb["minutes"], "tags": nb["tags"],
        "repo": REPO, "portfolio": PORTFOLIO, "file": nb["path"] + ".py",
        "source": f"{REPO}/blob/main/{nb['path']}.py",
        "raw": f"https://raw.githubusercontent.com/{REPO.split('github.com/')[1]}/main/{nb['path']}.py",
        "related": related, "assets": "../../assets/",
    }
    head = (f'<!--qn-theme--><link rel="stylesheet" href="../../assets/qn.css?v={VER}">'
            f'<link rel="stylesheet" href="../../assets/nb.css?v={VER}">'
            f'<meta name="description" content="{esc(nb["desc"])}">'
            f'<script>window.QN_NB={json.dumps(meta, ensure_ascii=False)}</script>'
            f'<script defer src="../../assets/nb.js?v={VER}"></script><!--/qn-theme-->')
    nb_id = nb["path"].removeprefix("notebooks/")
    doc = doc.replace("</head>", head + head_extra("../../", nb_id) + "</head>", 1)
    doc = re.sub(r"<title>.*?</title>", f"<title>{esc(nb['title'])} · Quant Notebooks</title>", doc, count=1, flags=re.S)
    f.write_text(doc, encoding="utf-8")
    print("tema aplicado:", f.relative_to(ROOT))

# ---------- home ----------
feat = next(n for n in nbs if n["id"] == CFG["featured"])
themes_used = [t for t in CFG["themes"] if any(n["theme"] == t for n in nbs)]


def norm(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


cards = []
for nb in nbs:
    search = norm(" ".join([nb["title"], nb["desc"], nb["theme"], nb["level"], *nb["tags"]]))
    href = nb["path"] + ".html"
    status = '<span class="draft">● em construção</span>' if nb.get("draft") else f'<span>{esc(nb["date"])}</span>'
    cards.append(f'''<a class="card" href="{href}" data-theme="{esc(nb['theme'])}" data-level="{esc(nb['level'])}" data-search="{esc(search)}">
<svg viewBox="0 0 392 150" data-kind="{nb['preview']}" aria-hidden="true"></svg>
<div style="margin-top:12px"><span class="chip on static">{esc(nb['theme'])}</span><span class="chip static">{esc(nb['level'])}</span></div>
<h3>{esc(nb['title'])}</h3><p>{esc(nb['desc'])}</p><div class="tags">{" · ".join(esc(t) for t in nb['tags'])}</div>
<div class="meta">{status}<span>◷ {nb['minutes']} min de leitura</span></div></a>''')

INDEX = f'''<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Notebooks · Marcelo Wizenberg</title>
<meta name="description" content="Notebooks educacionais de pesquisa quantitativa sobre B3, macro e indicadores, com código aberto em Python (marimo).">
<link rel="canonical" href="https://celowiz.github.io/quant-notebooks/">
<meta property="og:type" content="website"><meta property="og:site_name" content="Marcelo Wizenberg"><meta property="og:title" content="Quant Notebooks · Marcelo Wizenberg"><meta property="og:description" content="Notebooks educacionais de pesquisa quantitativa sobre B3, macro e indicadores, com código aberto em Python (marimo)."><meta property="og:url" content="https://celowiz.github.io/quant-notebooks/"><meta property="og:image" content="https://celowiz.github.io/quant-notebooks/og.png"><meta property="og:image:type" content="image/png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta property="og:image:alt" content="Quant Notebooks · celowiz.github.io/quant-notebooks"><meta property="og:locale" content="pt_BR">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:creator" content="@marcelo_wz"><meta name="twitter:title" content="Quant Notebooks · Marcelo Wizenberg"><meta name="twitter:description" content="Notebooks educacionais de pesquisa quantitativa sobre B3, macro e indicadores, com código aberto em Python (marimo)."><meta name="twitter:image" content="https://celowiz.github.io/quant-notebooks/og.png"><meta name="twitter:image:alt" content="Quant Notebooks · celowiz.github.io/quant-notebooks">
<link rel="stylesheet" href="assets/qn.css?v={VER}"><link rel="stylesheet" href="assets/home.css?v={VER}">{head_extra("", "home")}
</head><body>
<header class="qn-nav"><a class="qn-logo" href="{PORTFOLIO}"><span class="mw">m<b>w</b></span><span class="hide-sm">Marcelo Wizenberg</span><span class="sep hide-sm">/</span><span class="cur">Quant Notebooks</span></a>
<nav class="qn-links"><a href="{PORTFOLIO}">Portfólio</a><a class="on" href="#galeria">Notebooks</a><a class="hide-sm" href="https://celowiz.github.io/monitor-b3/">Monitor B3</a><a href="{REPO}" target="_blank" rel="noopener">GitHub ↗</a></nav></header>
<section class="hero"><canvas id="hero-bg" aria-hidden="true"></canvas>
<div class="hero-in"><div><div class="eyebrow">Pesquisa quantitativa · aberta</div>
<h1>Notebooks de mercado,<br><i>do dado ao gráfico.</i></h1>
<p>Estudos reproduzíveis sobre B3, macro e indicadores — com código, explicações e gráficos. Escritos em Python com marimo, para aprender fazendo.</p>
<div class="stats"><div><b>{len(nbs)}</b>notebooks</div><div><b>{len(themes_used)}</b>temas</div><div><b>Python</b>marimo · pandas</div></div></div>
<a class="feat" href="{feat['path']}.html"><div class="eyebrow">★ Em destaque</div><h2>{esc(feat['title'])}</h2><p style="color:var(--mut)">{esc(feat['desc'])}</p>
<svg id="feat-svg" viewBox="0 0 426 230" aria-hidden="true"></svg><div class="ill" style="margin-top:6px">gráfico ilustrativo</div>
<div style="margin-top:14px"><span class="btn p">Abrir notebook →</span></div></a></div></section>
<main class="wrap" id="galeria"><div class="ghead"><div><div class="eyebrow">Galeria</div><h2>Todos os notebooks</h2></div>
<div class="filters"><div class="row"><button class="chip on" data-theme-f="Todos">Todos</button>{"".join(f'<button class="chip" data-theme-f="{esc(t)}">{esc(t)}</button>' for t in themes_used)}</div>
<div class="row">{"".join(f'<button class="chip" data-level-f="{esc(l)}">{esc(l)}</button>' for l in CFG["levels"])}<input id="q" class="search" type="search" placeholder="⌕ Buscar…" aria-label="Buscar notebooks"></div></div></div>
<div class="grid">{"".join(cards)}</div><p class="empty" id="empty">Nenhum notebook com esses filtros.</p>
<div class="ill" style="margin-top:14px">miniaturas ilustrativas · os gráficos reais estão dentro de cada notebook</div></main>
<footer><p class="disclaimer">Conteúdo educacional. Nada aqui é recomendação de investimento.<br>Dados de fontes públicas (B3, Banco Central, Yahoo Finance) podem conter atrasos ou erros.</p>
<p class="disclaimer"><a href="{PORTFOLIO}">← celowiz.github.io</a> · <a href="{REPO}">código-fonte</a></p></footer>
<script src="assets/previews.js?v={VER}"></script><script src="assets/home.js?v={VER}"></script>
</body></html>
'''
(ROOT / "index.html").write_text(INDEX, encoding="utf-8")
print("index.html gerado com", len(nbs), "notebooks")
