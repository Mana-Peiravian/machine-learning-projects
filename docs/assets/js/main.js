// Central configuration: remote-derived defaults and optional placeholders.
const portfolioConfig = {
  githubUsername: "Mana-Peiravian",
  repositoryName: "machine-learning-projects",
  repositoryUrl: "https://github.com/Mana-Peiravian/machine-learning-projects",
  branch: "main",
  authorName: "ADD_APPROVED_PUBLIC_NAME",
  university: "ADD_UNIVERSITY",
  department: "ADD_DEPARTMENT",
  publicEmail: "ADD_APPROVED_PUBLIC_EMAIL",
  linkedinUrl: "ADD_LINKEDIN_URL"
};
// Optional placeholders are not rendered as verified facts.
document.querySelectorAll('[data-repository]').forEach(a => {a.href = portfolioConfig.repositoryUrl;});
document.querySelectorAll('[data-repo-path]').forEach(a => {
  a.href = portfolioConfig.repositoryUrl + '/' + (a.dataset.repoKind || 'blob') + '/' + encodeURIComponent(portfolioConfig.branch) + '/' + a.dataset.repoPath;
});
document.querySelectorAll('[data-site-home]').forEach(a => {
  a.href = 'https://' + portfolioConfig.githubUsername.toLowerCase() + '.github.io/' + encodeURIComponent(portfolioConfig.repositoryName) + '/';
});
const toggle = document.getElementById('theme-toggle');
const root = document.documentElement;
function isDark() {return root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;}
function updateLabel() {
  toggle.textContent = isDark() ? 'Light theme' : 'Dark theme';
  toggle.setAttribute('aria-label', isDark() ? 'Switch to light theme' : 'Switch to dark theme');
}
if (toggle) {
  try {const saved = localStorage.getItem('ml-portfolio-theme'); if (saved === 'dark' || saved === 'light') root.dataset.theme = saved;} catch (_) {}
  toggle.hidden = false; updateLabel();
  toggle.addEventListener('click', () => {
    root.dataset.theme = isDark() ? 'light' : 'dark';
    try {localStorage.setItem('ml-portfolio-theme', root.dataset.theme);} catch (_) {}
    updateLabel();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateLabel);
}
