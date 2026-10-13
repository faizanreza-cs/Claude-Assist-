# GitHub profile cleanup checklist

An audit of github.com/faizanreza-cs as of October 2026, with the steps to make it read as an
**AI engineering + pharma analytics** portfolio. These are changes only you can make in
GitHub settings. Nothing here has been deleted or changed for you.

## Where the profile stands

- 38 public repos, **34 of them forks**. Recruiters and hiring managers skip forks, and a page of
  them buries your own work.
- Only 4 original repos, none substantial yet: the PharmaPulse repo,
  `greyatom-python-for-data-science` (2020 coursework), `codersrank-practice`,
  `friendly-octo-meme` (empty, 2017).
- No profile README, a generic bio, no pinned repos, no topics.

## 1. Feature the flagship (15 minutes)

- [ ] **Rename** the PharmaPulse repo to `pharmapulse` (Settings → General → Repository name)
      *before* merging. The README and profile links already point at the new name, and GitHub
      redirects the old URL.
- [ ] Merge the PharmaPulse branch into `main` with **Squash and merge**, so the README, badges
      and screenshots show on the repo page. In the merge dialog, delete any auto-added
      `Co-authored-by` lines from the commit message. Then delete the feature branch.
- [ ] Set the repo description: *"AI analytics copilot for pharma brand teams: LLM agent, forecasting, patient persistence, HCP targeting"*.
- [ ] Add topics: `pharma-analytics` `llm-agents` `llm` `forecasting` `survival-analysis`
      `healthcare-analytics` `fastapi` `streamlit` `duckdb` `python`.
- [ ] Turn on GitHub Actions for the repo and confirm the CI badge goes green.

## 2. Profile README (10 minutes)

- [ ] Create a **public** repo named exactly `faizanreza-cs`, and copy `docs/profile/README.md` into it as `README.md`.
- [ ] Fill in the LinkedIn TODO (and a portfolio or blog link if you have one).

## 3. Profile settings (10 minutes)

- [ ] **Bio** (≤160 chars): *AI Engineer · LLM agents, forecasting & patient analytics for pharma commercial teams · Python · SQL · LLMs*
- [ ] Name: Faizan Reza · Location: Delhi, India · enable "Available for hire" (already on).
- [ ] Add LinkedIn and, if you have one, a website or blog under Social accounts.
- [ ] Use a clear, professional profile photo.
- [ ] **Pin** (Customize your pins): PharmaPulse first, then each new original project as you ship it. Pin 4–6 repos maximum, no forks.

## 4. Remove fork noise (20 minutes)

Make these **private** (Settings → Danger Zone → Change visibility) or delete them, unless you
committed your own work to one. Use ⭐ stars to bookmark references instead of forking.

**Interview prep, cheat sheets, link lists** (no signal for an AI or pharma role):
`tech-interview-handbook`, `coding-interview-university`, `cracking-the-data-science-interview`,
`Front-end-Developer-Interview-Questions`, `33-js-concepts`, `algodeck`, `40-Algorithms-Every-Programmer-Should-Know`,
`python-cheatsheet`, `Data-Science--Cheat-Sheet`, `Best-websites-a-programmer-should-visit`,
`project-based-learning`, `awesome-remote-job`, `remote-jobs`, `awesome-pretrained-stylegan`

**Upstream code you didn't change**:
`Python`, `go`, `standard`, `models`, `altair`, `datasets`, `InstaPy`, `ALAE`, `3d-photo-inpainting`,
`python-scripts`, `data-science-ipython-notebooks`, `covid19india-cluster`, `covid-dashboard`

**Recent AI forks.** Relevant, but still forks. Keep one only if you turn it into your own
work (a write-up, an extension, a benchmark), otherwise star it and remove the fork:
`MiroFish`, `Machine-learning-System-form-harvard`, `AI-Agents-Projects-Repo`, `real-world-llm-agents`,
`llm-career-roadmap-`, `Code-security-scanner`, `rag-cookbooks`

**Old originals**:
- [ ] Delete `friendly-octo-meme` (empty).
- [ ] `greyatom-python-for-data-science`: add a short README on what you learned, or archive it.
- [ ] `codersrank-practice`: archive it, or make it private.

Also check the `Rezabsinfotech` organization (`ALAE` fork, `cuddy-oenta`). It appears on your
profile, so archive or make private anything that isn't meant to be shown.

## 5. Next 90 days: two more original projects

Hiring managers look for **2–3 original, finished projects with clear READMEs and steady commit
history**. Suggested next builds, each covering a different part of the pharma value chain:

1. **Drug-safety signal detection** (pharmacovigilance + RAG): pull openFDA FAERS adverse
   event reports, compute disproportionality (PRR, ROR with confidence intervals), and add a
   LLM assistant that answers "is X a known side effect of Y?" with citations to FDA labels.
2. **Clinical-trial intelligence** (R&D analytics): ingest ClinicalTrials.gov, use an LLM to
   extract structured eligibility criteria, and forecast enrollment duration by phase, area and site count.
3. For each one, publish a short write-up (LinkedIn post, Medium or dev.to) with the key chart
   and a link to the repo.

Commit small and often (daily or weekly). A steady contribution graph backed by real commits
says more than bursts.

## 6. Resume and LinkedIn bullets from PharmaPulse

Use these as written. Every number is reproducible from the repo.

- Built an LLM analytics copilot (provider-agnostic tool-use agent on the Anthropic API) that answers pharma
  brand questions by calling 12 tested analytics tools, with every figure traced to its source
  and model-written SQL constrained by a sqlglot read-only guard.
- Developed weekly TRx/NRx forecasting (Holt-Winters, gradient boosting, seasonal baseline)
  with automatic model selection by holdout WAPE: 3.1% mean WAPE across four brands.
- Implemented patient-level persistence (Kaplan-Meier), PDC adherence and switching analytics on
  ~120k pharmacy claims, plus HCP decile/segment targeting and adstocked promotional response.
- Shipped it as a FastAPI service and Streamlit dashboard with Docker, GitHub Actions CI,
  63 tests and a 25-question golden evaluation harness.
