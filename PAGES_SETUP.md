# GitHub Pages setup

The complete site lives inside docs/; no backend, install or build is needed. Coursework outside docs/ is linked on GitHub. No PDFs or datasets are copied to the site.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open http://localhost:8000/ and stop with Ctrl+C. Test project pages and theme selection. The only stored preference is local theme; no trackers or third-party scripts are used.

## Activate Pages

1. Complete [publication review](PUBLICATION_REVIEW.md): identifiers, consent, course policy and data rights. Ignore rules do not remove tracked content/history.
2. Confirm repository values and publishing branch in portfolioConfig at the top of docs/assets/js/main.js; update HTML fallback links if these change.
3. Review, commit and push generated files.
4. Open **GitHub repository → Settings → Pages → Build and deployment**.
5. Choose **Source: Deploy from a branch**.
6. Select **main** and **/docs**, then **Save**.
7. Wait for deployment, confirm the displayed URL and visit [https://mana-peiravian.github.io/machine-learning-projects/](https://mana-peiravian.github.io/machine-learning-projects/).

These follow [GitHub publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site). Hosting was not enabled here. Confirm account visibility/hosting options: do not assume a private repository guarantees a private Pages website.

## Configuration

Remote-derived username/repository/URL use main. University, department, public name/contact and LinkedIn have optional placeholders. Update HTML fallback links, README, citation template and this document if publishing elsewhere.

The 404 page has self-contained inline styles and absolute homepage links for arbitrary missing nested URLs. Normal pages/assets use relative Project Pages paths. No sitemap/canonical URL is asserted before deployment.

## Recommended Git commands

```bash
git status
git diff --stat
git diff
git add ".gitignore" ".gitignore.portfolio-suggestions" ".portfolio/original-files.json" ".portfolio/validate.py" "1/README.generated.md" "2/README.md" "3/README.md" "AGENTS.md" "CITATION.template.cff" "CONTRIBUTING.md" "LICENSE-NOTICE.md" "PAGES_SETUP.md" "PORTFOLIO_AUDIT.md" "PORTFOLIO_GENERATION_REPORT.md" "PUBLICATION_REVIEW.md" "README.md" "docs/.nojekyll" "docs/404.html" "docs/assets/css/style.css" "docs/assets/js/main.js" "docs/favicon.svg" "docs/index.html" "docs/projects/imbalanced-ensembles.html" "docs/projects/linear-game-agents.html" "docs/projects/real-world-kernel-study.html"
git diff --cached --stat
git diff --cached
git commit -m "Add course portfolio documentation and GitHub Pages site"
git push origin main
```
