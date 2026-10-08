"""Read-only stdlib validator for portfolio links and coursework integrity."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib, json, re
ROOT = Path(__file__).resolve().parents[1]

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links, self.ids, self.tags, self.stack, self.errors = [], [], [], [], []
        self.lang = self.viewport = self.title = False
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append(tag)
        if 'id' in a: self.ids.append(a['id'])
        for k in ('href', 'src'):
            if k in a: self.links.append(a[k])
        if tag == 'html': self.lang = a.get('lang') == 'en'
        if tag == 'meta' and a.get('name') == 'viewport': self.viewport = True
        if tag == 'title': self.title = True
        if tag == 'img' and not a.get('alt'): self.errors.append('Missing alt text')
        if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:
            self.stack.append(tag)
    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag: self.errors.append('Unbalanced tag: ' + tag)
        else: self.stack.pop()

def validate():
    errors = []
    baseline = json.loads((ROOT/'.portfolio/original-files.json').read_text(encoding='utf-8'))
    for item in baseline:
        p = ROOT/item['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            errors.append('Original changed/missing: ' + item['path'])
    originals = {x['path'] for x in baseline}
    markdown = [p for p in ROOT.rglob('*.md') if '.git' not in p.parts and p.relative_to(ROOT).as_posix() not in originals]
    pages = {p: Page(p.read_text(encoding='utf-8')) for p in (ROOT/'docs').rglob('*.html')}
    links = []
    for p, page in pages.items():
        if len(page.ids) != len(set(page.ids)): errors.append('Duplicate IDs: ' + str(p))
        if not (page.lang and page.viewport and page.title and 'nav' in page.tags and 'main' in page.tags):
            errors.append('Missing metadata/landmarks: ' + str(p))
        errors.extend(str(p) + ': ' + e for e in page.errors)
        if page.stack: errors.append('Unclosed tags: ' + str(p))
        links.extend((p, u) for u in page.links)
    for p in markdown:
        fence = chr(96)*3
        text = re.sub(fence+r'.*?'+fence, '', p.read_text(encoding='utf-8'), flags=re.S)
        links.extend((p, u) for u in re.findall(r'!?\[[^\]]*\]\(([^)\n]+)\)', text))

    def exact_path(source, url, target):
        try: parts = target.relative_to(ROOT).parts
        except ValueError:
            errors.append('Escapes repository: ' + url)
            return
        cursor = ROOT
        for part in parts:
            if not cursor.is_dir() or part not in {x.name for x in cursor.iterdir()}:
                errors.append(str(source.relative_to(ROOT)) + ' missing/case mismatch: ' + url)
                return
            cursor /= part

    for source, url in links:
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc:
            marker = 'https://github.com/Mana-Peiravian/machine-learning-projects/'
            if url.startswith(marker) and '/main/' in url:
                exact_path(source, url, (ROOT/unquote(url.split('/main/', 1)[1])).resolve())
            continue
        if parsed.path.startswith('/'): errors.append('Root-relative path: ' + url)
        target = (source.parent/unquote(parsed.path)).resolve() if parsed.path else source
        exact_path(source, url, target)
        if source.suffix == '.html' and not target.is_relative_to(ROOT/'docs'):
            errors.append('HTML target outside docs: ' + url)
        if parsed.fragment and target in pages and unquote(parsed.fragment) not in pages[target].ids:
            errors.append('Missing fragment: ' + url)
    for p in pages:
        for path in re.findall(r'data-repo-path="([^"]+)"', p.read_text(encoding='utf-8')):
            exact_path(p, path, (ROOT/unquote(path)).resolve())
    for p in (ROOT/'docs').rglob('*.css'):
        for u in re.findall(r'url\(["\']?([^)"\']+)', p.read_text(encoding='utf-8')):
            if not urlsplit(u).scheme:
                exact_path(p, u, (p.parent/u).resolve())
    summary = dict(original_files_checked=len(baseline), html_pages_checked=len(pages),
                   markdown_files_checked=len(markdown), links_checked=len(links), errors=errors)
    print(json.dumps(summary, indent=2))
    if errors: raise SystemExit(1)
if __name__ == '__main__':
    validate()
