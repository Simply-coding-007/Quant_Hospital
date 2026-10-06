"""Helper utility to push the repository to GitHub remote."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import dulwich.porcelain as porcelain
from dulwich.repo import Repo

REMOTE_URL = "https://github.com/Simply-coding-007/Quant_Hospital.git"


def push_repo(token: str | None = None) -> None:
    repo_path = Path(__file__).resolve().parent
    repo = Repo(str(repo_path))

    target_url = REMOTE_URL
    if token:
        # Insert token into https URL
        target_url = REMOTE_URL.replace("https://", f"https://{token}@")
    elif "GITHUB_TOKEN" in os.environ:
        tok = os.environ["GITHUB_TOKEN"].strip()
        target_url = REMOTE_URL.replace("https://", f"https://{tok}@")
    elif "GH_TOKEN" in os.environ:
        tok = os.environ["GH_TOKEN"].strip()
        target_url = REMOTE_URL.replace("https://", f"https://{tok}@")

    print(f"Staging latest files in {repo_path}...")
    porcelain.add(
        repo,
        paths=[
            ".gitignore",
            "README.md",
            "requirements.txt",
            "app.py",
            "generate_plots.py",
            "push_to_github.py",
            "src",
            "models",
            "results",
        ],
    )

    try:
        commit_id = porcelain.commit(
            repo,
            message="Release v1.0.0: Quantum-Secure Biomedical Network Layer".encode("utf-8"),
            author="Senior Engineering Team <team@quanthospital.ai>".encode("utf-8"),
        )
        print(f"Committed new changes: {commit_id.decode('ascii')}")
    except Exception as e:
        print(f"No new changes to commit: {e}")

    print(f"Pushing to GitHub remote: {REMOTE_URL}...")
    try:
        porcelain.push(repo, target_url, refspecs=["refs/heads/master:refs/heads/main"])
        print(" Successfully pushed to main branch on GitHub!")
    except Exception as e1:
        try:
            porcelain.push(repo, target_url, refspecs=["refs/heads/master:refs/heads/master"])
            print(" Successfully pushed to master branch on GitHub!")
        except Exception as e2:
            print(f" Push failed: {e2}")
            print("\nTo push with authentication, run:")
            print("  python push_to_github.py --token <YOUR_GITHUB_PERSONAL_ACCESS_TOKEN>")
            print("or set the environment variable GITHUB_TOKEN.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Quant_Hospital to GitHub.")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token")
    args = parser.parse_args()
    push_repo(token=args.token)
