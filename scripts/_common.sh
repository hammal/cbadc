# Shared setup for the scripts/ entry points.
#
# These scripts are the single definition of what "lint", "test" and
# "performance" mean for cbadc. Both CI systems call them, so a command only
# ever needs changing in one place:
#
#   .github/workflows/  -> public validation, releases, docs, containers
#   .gitlab-ci.yml      -> private development validation
#
# Sourced, not executed. Do not add `set -e` here; each caller sets its own.

# Headless by default: the suite creates a lot of figures and an interactive
# matplotlib backend is both slower and a hard failure on bare CI runners.
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Resolve the repository root from this file's location so the scripts work
# regardless of the caller's working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# All commands run through `uv`, matching how the project is installed
# (`uv sync --extra dev --extra delsig`). Fail with a useful message rather than
# "command not found" if it is missing.
if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is not on PATH." >&2
    echo "       cbadc is managed with uv: https://docs.astral.sh/uv/" >&2
    exit 127
fi

# Echo each command before running it, so CI logs show exactly what ran.
run() {
    echo "+ $*" >&2
    "$@"
}

# Run a tool from the dev environment.
#
# A bare `uv run` re-syncs to core dependencies and removes extras. Tests need
# both dev tools and the delta-sigma bridge, so request them on every run.
uvrun() {
    run uv run --extra dev --extra delsig "$@"
}
