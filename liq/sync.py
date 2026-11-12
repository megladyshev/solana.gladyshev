"""Two-way sync with the `liq-data` branch on GitHub, the meeting point of the two collectors:
    pc/     what this computer recorded (Binance, Bybit, OKX in real time, OI): pushed from here
    cloud/  OKX's last ~24 h fetched every 6 h by GitHub Actions (.github/workflows/liq-backfill.yml): pulled here

The branch is checked out in data/liq-repo (ignored by the main repo). The collector runs this every 6 h;
it is safe to run by hand:  python liq/sync.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from merge import merge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / "data" / "liq"
REPO = ROOT / "data" / "liq-repo"
BRANCH = "liq-data"


def git(*args, cwd=REPO, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=180)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[-300:]}")
    return r


def same_author():
    for k in ("user.name", "user.email"):                 # the main repo's author
        git("config", k, git("config", k, cwd=ROOT).stdout.strip())


def ensure_repo():
    if (REPO / ".git").exists():
        return
    url = git("remote", "get-url", "origin", cwd=ROOT).stdout.strip()
    if git("clone", "--quiet", "--branch", BRANCH, "--single-branch", url, str(REPO), cwd=ROOT, check=False).returncode == 0:
        return
    REPO.mkdir(parents=True, exist_ok=True)               # first run anywhere: start the branch
    git("init", "--quiet", "-b", BRANCH)
    git("remote", "add", "origin", url)
    same_author()
    (REPO / "README.md").write_text("Liquidation recordings. pc/ = the home computer's live feeds, cloud/ = OKX history from GitHub Actions.\n"
                                    "Daily CSV per kind; merge with liq/merge.py, read with liq/summary.py.\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "--quiet", "-m", "Start the liquidation data branch")


def main():
    ensure_repo()
    same_author()
    if git("ls-remote", "--exit-code", "--heads", "origin", BRANCH, check=False).returncode == 0:
        git("pull", "--quiet", "--rebase", "origin", BRANCH)
    n = merge(LOCAL, REPO / "pc")
    git("add", "-A")
    if git("diff", "--cached", "--quiet", check=False).returncode:
        git("commit", "--quiet", "-m", f"pc: {n} file(s)")
    for _ in range(3):                                    # the cloud job may have pushed meanwhile
        if git("push", "--quiet", "-u", "origin", BRANCH, check=False).returncode == 0:
            break
        git("pull", "--quiet", "--rebase", "origin", BRANCH)
    else:
        raise RuntimeError("push failed 3 times")
    cloud = len(list((REPO / "cloud").glob("*.csv"))) if (REPO / "cloud").exists() else 0
    print(f"synced: {n} pc file(s) updated, {cloud} cloud file(s) available")


if __name__ == "__main__":
    main()
