"""
=============================================================================
sync.py — GitHub Push/Pull Synchronization for Territorial.io AI
=============================================================================

Handles all GitHub operations to keep 6 training sessions synchronized.
Each session pushes its data to GitHub; other sessions pull the latest 
master model before starting.

Works identically on both Kaggle and Google Colab free tiers.
Reads GITHUB_TOKEN from os.environ or Kaggle secrets.

Repo: https://github.com/amer6767/Te-ai

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import os
import json
import time
import base64
import requests
from typing import Optional, Callable

# ==============================================
# CONFIGURATION
# ==============================================

DEFAULT_REPO_URL = "https://github.com/amer6767/Te-ai"
DEFAULT_BRANCH = "main"
LOCK_FILE = ".sync_lock"
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5
RATE_LIMIT_DELAY = 60  # Wait 60 seconds on rate limit


# ==============================================
# GITHUB SYNC CLASS
# ==============================================

class GitHubSync:
    """
    Synchronizes training data between 6 cloud sessions via GitHub.
    
    Each Colab/Kaggle session pushes its session file and pulls the
    master model. Uses GitHub's REST API (no git CLI needed).
    
    Features:
        - Lock file mechanism to prevent concurrent pushes
        - Automatic retry on rate limiting
        - Works with both os.environ GITHUB_TOKEN and Kaggle secrets
        - Unique filenames per session so pushes never overwrite
    
    Usage:
        sync = GitHubSync()
        sync.push_session("colab1")       # Push session data
        sync.pull_master()                 # Pull master model
        sync.push_rated_moves()            # Push human ratings
        sync.pull_review_queue()           # Pull moves needing review
    """

    def __init__(self, repo_url: str = DEFAULT_REPO_URL, token: Optional[str] = None):
        """
        Initialize GitHub sync.
        
        Args:
            repo_url: GitHub repository URL
            token:    GitHub personal access token. If None, reads from
                      GITHUB_TOKEN env var or Kaggle secrets.
        """
        self.repo_url = repo_url
        self.branch = DEFAULT_BRANCH

        # Parse owner/repo from URL
        parts = repo_url.rstrip("/").split("/")
        self.owner = parts[-2]
        self.repo = parts[-1]

        # Get token
        self.token = token or self._get_token()

        # Set up API headers
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        self.api_base = f"https://api.github.com/repos/{self.owner}/{self.repo}"

        if self.token:
            print(f"   🔑 GitHub token loaded (repo: {self.owner}/{self.repo})")
        else:
            print(f"   ⚠️ No GitHub token found — sync will be read-only")

    def _get_token(self) -> Optional[str]:
        """
        Get GitHub token from environment or Kaggle secrets.
        
        Tries in order:
            1. GITHUB_TOKEN environment variable
            2. Kaggle secrets (via kaggle_secrets UserSecretsClient)
            3. GH_TOKEN environment variable (alternative name)
        """
        # Try standard environment variable
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            return token

        # Try Kaggle secrets
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
            token = client.get_secret("GITHUB_TOKEN")
            if token:
                return token
        except (ImportError, Exception):
            pass

        # Try alternative env var name
        token = os.environ.get("GH_TOKEN")
        if token:
            return token

        return None

    def push_session(self, session_name: str) -> bool:
        """
        Push a session file to GitHub.
        
        Uses a unique filename so it never overwrites other sessions.
        
        Args:
            session_name: e.g., "colab1", "kaggle2"
            
        Returns:
            True if push succeeded
        """
        filepath = f"session_{session_name}.json"

        # Find the session file locally
        local_path = filepath
        if not os.path.exists(local_path):
            local_path = os.path.join("sessions", filepath)
        if not os.path.exists(local_path):
            print(f"   ❌ Session file not found: {filepath}")
            return False

        return self._retry(lambda: self._push_file(local_path, filepath))

    def pull_master(self) -> bool:
        """
        Pull master_model.json from GitHub to local disk.
        
        Returns:
            True if pull succeeded
        """
        return self._retry(lambda: self._pull_file("master_model.json", "master_model.json"))

    def push_rated_moves(self) -> bool:
        """
        Push rated_moves.json to GitHub.
        
        Returns:
            True if push succeeded
        """
        if not os.path.exists("rated_moves.json"):
            print("   ⚠️ No rated_moves.json found locally")
            return False

        return self._retry(lambda: self._push_file("rated_moves.json", "rated_moves.json"))

    def pull_review_queue(self) -> bool:
        """
        Pull review_queue.json from GitHub to local disk.
        
        Returns:
            True if pull succeeded
        """
        return self._retry(lambda: self._pull_file("review_queue.json", "review_queue.json"))

    def _push_file(self, local_path: str, remote_path: str) -> bool:
        """
        Push a file to the GitHub repo via API.
        
        Args:
            local_path:  Path to local file
            remote_path: Path in the repo
            
        Returns:
            True if push succeeded
        """
        if not self.token:
            print(f"   ❌ Cannot push: no GitHub token")
            return False

        # Acquire lock
        if not self._acquire_lock():
            print(f"   ❌ Could not acquire lock — another session may be pushing")
            return False

        try:
            # Read local file
            with open(local_path, 'rb') as f:
                content = f.read()

            # Encode to base64
            content_b64 = base64.b64encode(content).decode('utf-8')

            # Check if file already exists (need SHA for update)
            url = f"{self.api_base}/contents/{remote_path}"
            existing = requests.get(url, headers=self.headers, params={"ref": self.branch})

            data = {
                "message": f"Update {remote_path} from training session",
                "content": content_b64,
                "branch": self.branch,
            }

            if existing.status_code == 200:
                # File exists — include SHA for update
                data["sha"] = existing.json()["sha"]

            response = requests.put(url, headers=self.headers, json=data)

            if response.status_code in [200, 201]:
                print(f"   ✅ Pushed {remote_path} to GitHub")
                return True
            elif response.status_code == 403:
                print(f"   ⚠️ Rate limited — waiting {RATE_LIMIT_DELAY}s...")
                time.sleep(RATE_LIMIT_DELAY)
                return False
            else:
                print(f"   ❌ Push failed ({response.status_code}): {response.text[:200]}")
                return False

        finally:
            self._release_lock()

    def _pull_file(self, remote_path: str, local_path: str) -> bool:
        """
        Pull a file from GitHub repo to local disk.
        
        Args:
            remote_path: Path in the repo
            local_path:  Path to save locally
            
        Returns:
            True if pull succeeded
        """
        url = f"{self.api_base}/contents/{remote_path}"
        params = {"ref": self.branch}

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            file_info = response.json()
            content = base64.b64decode(file_info["content"])

            # Ensure directory exists
            dir_name = os.path.dirname(local_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(local_path, 'wb') as f:
                f.write(content)

            print(f"   ✅ Pulled {remote_path} from GitHub")
            return True

        elif response.status_code == 404:
            print(f"   ⚠️ {remote_path} not found on GitHub")
            return False

        elif response.status_code == 403:
            print(f"   ⚠️ Rate limited — waiting {RATE_LIMIT_DELAY}s...")
            time.sleep(RATE_LIMIT_DELAY)
            return False

        else:
            print(f"   ❌ Pull failed ({response.status_code}): {response.text[:200]}")
            return False

    def _acquire_lock(self) -> bool:
        """
        Create a lock file on GitHub to prevent concurrent pushes.
        
        Returns:
            True if lock was acquired successfully
        """
        if not self.token:
            return True  # No token → read-only → no lock needed

        url = f"{self.api_base}/contents/{LOCK_FILE}"
        lock_content = json.dumps({
            "locked_by": os.environ.get("HOSTNAME", "unknown"),
            "locked_at": time.time(),
        })

        data = {
            "message": "Acquire sync lock",
            "content": base64.b64encode(lock_content.encode()).decode('utf-8'),
            "branch": self.branch,
        }

        # Check if lock already exists
        existing = requests.get(url, headers=self.headers, params={"ref": self.branch})
        if existing.status_code == 200:
            # Lock exists — check if it's stale (older than 5 minutes)
            try:
                lock_info = json.loads(base64.b64decode(existing.json()["content"]))
                age = time.time() - lock_info.get("locked_at", 0)
                if age < 300:  # Less than 5 minutes
                    print(f"   ⏳ Lock held by {lock_info.get('locked_by', 'unknown')} "
                          f"for {age:.0f}s — waiting...")
                    return False
                # Stale lock — override it
                data["sha"] = existing.json()["sha"]
            except Exception:
                data["sha"] = existing.json()["sha"]

        response = requests.put(url, headers=self.headers, json=data)
        return response.status_code in [200, 201]

    def _release_lock(self):
        """Delete the lock file from GitHub after pushing."""
        if not self.token:
            return

        url = f"{self.api_base}/contents/{LOCK_FILE}"
        existing = requests.get(url, headers=self.headers, params={"ref": self.branch})

        if existing.status_code == 200:
            data = {
                "message": "Release sync lock",
                "sha": existing.json()["sha"],
                "branch": self.branch,
            }
            requests.delete(url, headers=self.headers, json=data)

    def _retry(self, func: Callable, max_attempts: int = MAX_RETRY_ATTEMPTS) -> bool:
        """
        Retry a function on failure with exponential backoff.
        
        Args:
            func:         Callable that returns True on success, False on failure
            max_attempts: Maximum number of retry attempts
            
        Returns:
            True if the function eventually succeeded
        """
        for attempt in range(1, max_attempts + 1):
            try:
                result = func()
                if result:
                    return True
                if attempt < max_attempts:
                    delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                    print(f"   🔄 Retry {attempt}/{max_attempts} in {delay}s...")
                    time.sleep(delay)
            except Exception as e:
                print(f"   ❌ Attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
                    time.sleep(delay)

        return False


# ==============================================
# DEMO
# ==============================================

def demo():
    """Quick demo — tests token loading and API connectivity."""
    print("\n🔄 GitHub Sync Demo")
    print("=" * 50)

    sync = GitHubSync()

    # Test API connectivity
    print("\n🧪 Testing GitHub API connection...")
    url = f"{sync.api_base}"
    try:
        response = requests.get(url, headers=sync.headers)
        if response.status_code == 200:
            repo_info = response.json()
            print(f"   ✅ Connected to: {repo_info.get('full_name', 'unknown')}")
            print(f"   ⭐ Stars: {repo_info.get('stargazers_count', 0)}")
            print(f"   📝 Description: {repo_info.get('description', 'N/A')}")
        else:
            print(f"   ⚠️ Could not connect: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    demo()
