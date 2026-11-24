# Development workflow

cbadc is developed in a **private** repository and published from it to a
**public** one. This document is the contract for how code moves between them.

> **The rule that matters most**
>
> A branch created from `develop` may be pushed to GitHub after reviewing its
> commits and diff. A branch created from `next` must first be reconstructed on
> a clean branch based on `github/develop`.
>
> `tools/publish-github-pr` does that reconstruction for you, and is the only
> supported way to publish.

---

## The two repositories

| | Private | Public |
|---|---|---|
| Host | Private GitLab (`origin`) | `github.com/hammal/cbadc` |
| Git remote | `origin` | `github` |
| Visibility | private | public |
| Purpose | day-to-day development, unreleased work, student contributions, integration testing | released and review-ready code, issues, pull requests, releases |
| CI | `.gitlab-ci.yml` | `.github/workflows/` |
| Publishes artefacts | **no** | PyPI, ghcr.io, documentation |

`origin` is the default push target:

```console
$ git config --get remote.pushDefault
origin
```

so a bare `git push` goes to **GitLab**, never to GitHub. Reaching GitHub always
requires naming it explicitly, which is deliberate.

Verify your remotes at any time:

```console
$ git remote -v
github  https://github.com/hammal/cbadc.git (fetch)
github  https://github.com/hammal/cbadc.git (push)
origin  <private GitLab URL> (fetch)
origin  <private GitLab URL> (push)
```

There is **no mirroring** between the two repositories, in either direction, by
design. Every public commit is placed there by a deliberate act.

---

## Branch roles

| Branch | Lives on | Based on | Purpose |
|---|---|---|---|
| `develop` | GitLab **and** GitHub | — | The public-ready integration branch. The default branch on both hosts. Everything here is safe to publish. |
| `next` | GitLab **only** | `develop` | Private integration. Unreleased work is combined and tested here before it is considered for `develop`. **Never pushed to GitHub.** |
| `feature/*` | GitLab only | `next` | New functionality. |
| `fix/*` | GitLab only | `develop` or `next` | Bug fixes. Base on `develop` if the fix should be publishable on its own; on `next` if it depends on unreleased work. |
| `public/*` | local / GitLab only | `github/develop` | Clean reconstruction branches, carrying no private history. Created for you by `tools/publish-github-pr` as `public/<name>`, then pushed to GitHub as `pr/<name>`. |
| `pr/*` | GitHub | `github/develop` | Temporary heads for GitHub pull requests. Created only by `tools/publish-github-pr`; delete after the PR merges. |

`develop` and `next` are **protected** on GitLab: force-push is disabled, and
only Maintainers may push or merge. Everyone else contributes through merge
requests.

### Choosing a base

Base on `develop` when the change stands alone — a bug fix, a documentation
correction, a self-contained feature. It can then be published directly.

Base on `next` when the change depends on unreleased work. It cannot be
published as-is; the publication tool will reconstruct it and warn you.

---

## Everyday development (GitLab)

```console
$ git checkout next
$ git pull
$ git checkout -b feature/my-thing next

# ... work, commit ...

$ make test                       # lint + doctests + unit tests
$ git push -u origin feature/my-thing
```

Then open a merge request on GitLab targeting `next` (or `develop` for a
standalone fix). The pipeline must be green before it can merge — the project
is configured with *only allow merge if pipeline succeeds*.

### Student contributions

Students work on the **private GitLab repository** and contribute via merge
requests. They do **not** need write access to the public GitHub repository, and
should not be given it unless there is a specific reason. Only maintainers
publish branches to GitHub.

---

## Publishing to GitHub

Publication is a deliberate, reviewed, one-branch-at-a-time operation performed
with `tools/publish-github-pr`.

```console
$ tools/publish-github-pr --name zoh-boundary-fix --from fix/zoh-boundary
```

What the tool does, in order:

1. Refuses to run if the working tree is dirty.
2. Verifies the `github` remote exists and points at `hammal/cbadc` — it will
   not publish to an unverified destination.
3. Fetches both remotes.
4. Selects the commits: everything on `--from` that is not in `github/develop`,
   or exactly the SHAs given to `--commits`.
5. **Audits for private ancestry** and warns if the source descends from `next`
   or if a selected commit is already on `next` or already public.
6. Creates a clean branch from `github/develop` and cherry-picks **only** the
   selected commits, so the published branch shares no history with any private
   branch.
7. Runs the complete unit test gate (`scripts/lint`, `scripts/test-doctest`,
   `scripts/test-unit`).
8. Prints the commit list, the file stat and the **full diff** that would become
   public.
9. Requires you to type the branch name — not `y` — to confirm.
10. Pushes **one** explicit refspec to **one** `pr/*` branch.
11. Optionally opens the pull request with `gh` (`--create-pr`).

Any failure along the way cleans up the local branch and publishes nothing.

Useful options:

- `--commits a1b2c3d,e4f5a6b` — publish an explicit subset instead of a whole
  branch. Use this when a branch mixes publishable and unreleased work.
- `--update` — update a `pr/*` branch that already exists (uses
  `--force-with-lease`, never plain `--force`).

### Before every public push

Read the diff the tool prints. In full. It is the last checkpoint before the
code is on the public internet, where it may be cached or indexed even if
deleted afterwards.

---

## Synchronisation

`develop` exists on both hosts and must not diverge.

**GitHub → GitLab** (after a PR merges publicly, or someone else contributes):

```console
$ git fetch github
$ git checkout develop
$ git merge --ff-only github/develop
$ git push origin develop
```

**GitLab → GitHub**: only through `tools/publish-github-pr` and a pull request.
Never by pushing `develop` directly.

**Keeping `next` current** after `develop` moves:

```console
$ git checkout next
$ git merge develop
$ git push origin next
```

---

## CI responsibilities

The commands themselves live in `scripts/` and are called by *both* systems, so
they cannot drift:

| Script | What it runs |
|---|---|
| `scripts/lint` | `ruff check` and `ruff format --check` on `src/cbadc` |
| `scripts/test-unit` | the unit suite, xdist-parallel, JUnit to `pytest-unittests.xml` |
| `scripts/test-doctest` | doctests in `src/cbadc`, JUnit to `pytest-doctests.xml` |
| `scripts/test-performance` | all ngspice circuit and noise validation cases |
| `scripts/test-all` | lint + doctests + unit tests — the merge gate |

`make help` lists the developer-facing wrappers.

### GitLab CI — private validation

Lint and format checks, doctests, the unit suite across Python 3.11–3.15,
merge-request validation for student contributions, and performance validation
on `next` and `develop`.

It publishes **nothing**.

### GitHub Actions — public validation and publication

Pull-request validation, `develop` validation, documentation builds, container
publishing to `ghcr.io`, and PyPI releases.

### The ngspice bench

`tests/ngspice` checks a broad set of simulation and noise features of `cbadc.Frontend`
against a circuit simulator: each test builds a netlist whose differential
equation *is* the state space, runs it in ngspice and compares decisions,
states, spectra or variances. Locally `make test-ngspice` uses an XSPICE-enabled
`ngspice` on `PATH`; `make test-ngspice-docker` builds the repository `Dockerfile`
(Debian trixie, ngspice with XSPICE) and runs the bench inside it.
`make test-ngspice-podman` builds and runs the same image with Podman.
`.github/workflows/ngspice.yml` builds the Dockerfile on every pull request
to `develop` or `main` and push to either branch; the base image and package
versions can change over time. GitLab installs the Debian package in its
`performance` stage for merge requests to those branches and integration
branch pushes. The seventh-order and repeated-seed cases run in the same
ngspice job on every merge request.

The test inventory, comparison method, dated results, and remaining gaps are
documented in [Validation against ngspice](source/ngspice-validation.rst).

### Release responsibility

There is exactly **one** authoritative public release pipeline:
`.github/workflows/pypi-deployment.yml`. Publishing a non-prerelease GitHub
Release automatically builds, checks and installs both the sdist and wheel,
then publishes those same artifacts to PyPI using `PYPI_API_TOKEN`.
The tag must match `v<version>` in `pyproject.toml`; prereleases do not publish.
There is no separate manual workflow or TestPyPI upload.

Do not add a PyPI, ghcr.io or documentation-publishing job to `.gitlab-ci.yml`.
Private container images, if ever built, stay in the GitLab registry and are
never pushed to the public GitHub registry.

---

## Forbidden commands

Never run any of these:

```console
# Publishes every branch, including private ones.
git push --mirror github
git push github --all
git push github --tags

# Discards other people's work without checking.
git push github --force <anything>
```

Also never:

- Configure push mirroring from GitLab to GitHub. Only explicitly selected
  references are ever published.
- Configure bidirectional mirroring between the two repositories.
- Push `next`, `feature/*` or any other private branch to `github`.
- Publish without reading the diff.

If you need to overwrite a public branch, use `--force-with-lease`, which
refuses when the remote has moved since you last fetched. `tools/publish-github-pr --update`
already does this.

### The safe habit

```console
$ git push                                   # -> GitLab (remote.pushDefault)
$ tools/publish-github-pr --name ... --from ...   # -> GitHub, reviewed
```

If you find yourself typing `git push github` by hand, stop and use the tool
instead.
