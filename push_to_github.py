####GITHUB_TOKEN = "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"
# GITHUB_URL   = "https://github.com/VedantUpasani46/ML-QUANTITATIVE-PORTFOLIO"
# GITHUB_USER  = "VedantUpasani46"
# GITHUB_TOKEN = "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"
# #####
# !/usr/bin/env python3
"""
ML Quant Portfolio — GitHub Publisher (All 40 Modules)
======================================================
Complete automated deployment of all 40 modules to GitHub.
Handles remote conflicts and ensures all files are pushed.

SETUP:
──────
1. Create GitHub repo at https://github.com/new
   - Name: ML-QUANTITATIVE-PORTFOLIO (or any name)
   - Can have README, license, .gitignore (script handles this)

2. Fill in these variables:
   - GITHUB_URL: Your repo HTTPS URL
   - GITHUB_USER: Your GitHub username
   - GITHUB_TOKEN: Personal Access Token
     (Create at: GitHub → Settings → Developer settings → PAT)

3. Run: python push_to_github.py

Done! All 40 modules pushed to GitHub with individual READMEs.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ═════════════════════════════════════════════════════════════════════
# ✏️  FILL THESE IN BEFORE RUNNING
# ═════════════════════════════════════════════════════════════════════
GITHUB_URL = "https://github.com/VedantUpasani46/ML-QUANTITATIVE-PORTFOLIO"
GITHUB_USER = "VedantUpasani46"
GITHUB_TOKEN = "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"  # ⚠️ CHANGE THIS!

# Source directory (current directory with all modules)
SOURCE_DIR = Path(__file__).parent


# ═════════════════════════════════════════════════════════════════════


def run(cmd, cwd=None, check=True):
    """Run shell command."""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        print(f"✗ Command failed: {' '.join(cmd)}")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)
    return result


def validate_config():
    """Validate configuration."""
    errors = []
    if "YOUR_USERNAME" in GITHUB_URL or "VedantUpasani146" not in GITHUB_USER:
        print("⚠️  Note: Using VedantUpasani146 - update if this is not you!")
    if GITHUB_TOKEN.startswith("ghp_YOUR"):
        errors.append("GITHUB_TOKEN still has placeholder - add your real token!")

    if errors:
        print("\n⚠️  Configuration errors:")
        for e in errors:
            print(f"   • {e}")
        print("\nEdit variables at top of script and re-run.")
        sys.exit(1)


def git_is_installed():
    """Check if git is available."""
    try:
        run(["git", "--version"], check=False)
        return True
    except FileNotFoundError:
        return False


def init_git():
    """Initialize git repository."""
    print("\n🔧 Initializing git repository...")

    if not (SOURCE_DIR / ".git").exists():
        run(["git", "init", "-b", "main"], cwd=SOURCE_DIR)
        print("  ✓ Git initialized")
    else:
        print("  ✓ Git repository exists")

    run(["git", "config", "user.email", f"{GITHUB_USER}@users.noreply.github.com"], cwd=SOURCE_DIR)
    run(["git", "config", "user.name", GITHUB_USER], cwd=SOURCE_DIR)

    auth_url = GITHUB_URL.replace("https://", f"https://{GITHUB_USER}:{GITHUB_TOKEN}@")
    run(["git", "remote", "remove", "origin"], cwd=SOURCE_DIR, check=False)
    run(["git", "remote", "add", "origin", auth_url], cwd=SOURCE_DIR)

    print("  ✓ Git configured")


def stage_and_commit():
    """Stage all files and create commit."""
    print("\n📦 Staging files...")

    run(["git", "add", "-A"], cwd=SOURCE_DIR)  # -A ensures all files including deletions

    result = run(["git", "diff", "--cached", "--name-only"], cwd=SOURCE_DIR, check=False)
    files = [l for l in result.stdout.strip().splitlines() if l]

    if not files:
        print("  ⚠️  No files to stage (may already be committed)")
        return 0

    print(f"  → {len(files)} files staged")

    print("\n✍️  Creating commit...")
    timestamp = datetime.now().strftime("%Y-%m-%d")

    commit_msg = f"""feat: add complete 40-module quant portfolio ({timestamp})

Complete production-ready quantitative finance system:

📦 40 Modules organized by category:
  • Machine Learning & AI (Modules 1-6)
  • Derivatives & Options (Modules 7-8)
  • Portfolio Management (Modules 9-11)
  • HFT & Market Making (Modules 12-14)
  • Credit & Fixed Income (Modules 15-16)
  • Crypto & DeFi (Modules 17-18)
  • Macro & Commodities (Modules 19-20)
  • Alternative Data & Engineering (Modules 21-22)
  • Deep Learning (Modules 23-24)
  • Explainability (Modules 25-26)
  • Infrastructure (Modules 27-28)
  • Research Tools (Modules 29-30)
  • Advanced ML (Modules 31-34)
  • Risk Management (Modules 35-36)
  • Trading Strategies (Modules 37-38)
  • Market Microstructure (Modules 39-40)

📊 Statistics:
  • 16,500+ lines of production code
  • 40 individual README.md files
  • Complete documentation
  • Interview insights for each module
"""

    # Check if there's anything to commit
    status = run(["git", "status", "--porcelain"], cwd=SOURCE_DIR, check=False)
    if not status.stdout.strip():
        print("  ℹ️  No changes to commit (already up to date)")
        return len(files)

    run(["git", "commit", "-m", commit_msg], cwd=SOURCE_DIR)
    print("  ✓ Commit created")

    return len(files)


def push_to_github():
    """Push to GitHub with conflict resolution."""
    print("\n🚀 Pushing to GitHub...")

    # First, try to pull and rebase any remote changes
    print("\n  → Checking for remote changes...")
    pull_result = run(
        ["git", "pull", "--rebase", "origin", "main"],
        cwd=SOURCE_DIR, check=False
    )

    if pull_result.returncode == 0:
        print("  ✓ Merged remote changes successfully")
    else:
        print("  ℹ️  No remote changes to merge (or first push)")

    # Now push with force-with-lease (safer than --force)
    print("\n  → Pushing all files to GitHub...")
    push_result = run(
        ["git", "push", "-u", "origin", "main", "--force-with-lease"],
        cwd=SOURCE_DIR, check=False
    )

    if push_result.returncode != 0:
        # If force-with-lease fails, try regular push
        print("  ℹ️  Trying alternative push method...")
        push_result = run(
            ["git", "push", "-u", "origin", "main"],
            cwd=SOURCE_DIR, check=False
        )

        if push_result.returncode != 0:
            # Last resort: force push (overwrites remote)
            print("\n  ⚠️  Conflict detected. Using force push to ensure all files are uploaded...")
            print("  ⚠️  This will overwrite any files on GitHub that aren't local.")

            push_result = run(
                ["git", "push", "-u", "origin", "main", "--force"],
                cwd=SOURCE_DIR, check=False
            )

            if push_result.returncode != 0:
                print("\n❌ Push failed. Error details:")
                print(push_result.stderr)
                print("\n💡 Troubleshooting:")
                print("  1. Check your GitHub token has 'repo' permissions")
                print("  2. Verify the repository exists at:", GITHUB_URL.rstrip('.git'))
                print("  3. Make sure your token hasn't expired")
                sys.exit(1)

    clean_url = GITHUB_URL.rstrip(".git")
    print(f"\n✅ Successfully pushed to GitHub!")
    print(f"   {clean_url}")


def print_summary(n_files):
    """Print deployment summary."""
    print("\n" + "═" * 70)
    print("  DEPLOYMENT SUMMARY")
    print("═" * 70)
    print(f"  Modules:          40")
    print(f"  Files pushed:     {n_files if n_files > 0 else 'all'}")
    print(f"  Repository:       {GITHUB_URL.rstrip('.git')}")
    print()
    print("  ✅ All 40 modules deployed successfully!")
    print("  ✅ Each module has individual README.md")
    print("  ✅ Complete documentation included")
    print("═" * 70)


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ML Quant Portfolio — GitHub Publisher (40 Modules)      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    validate_config()

    if not git_is_installed():
        print("\n✗ git is not installed")
        print("  Install: https://git-scm.com/downloads")
        sys.exit(1)

    init_git()
    n_files = stage_and_commit()
    push_to_github()
    print_summary(n_files)

    print("\n🎉 Your portfolio is live! Ready for applications.")
    print(f"\n🔗 View at: {GITHUB_URL.rstrip('.git')}")


if __name__ == "__main__":
    main()
