"""
=============================================================================
run_session.py — Single-File Session Launcher for Territorial.io AI
=============================================================================

This is THE file to run on each Colab/Kaggle account to start training.

What it does:
1. Detects which account this is (from hostname or asks user)
2. Installs Playwright if not installed
3. Pulls latest master_model.json from GitHub
4. Loads master model weights into brain.py's GameAgent
5. Launches Playwright browser via game_environment.py
6. Runs 10 games via trainer.py
7. Saves session file
8. Exports unrated moves to review_queue.json
9. Pushes session file to GitHub

Usage on Colab:
    !pip install playwright
    !playwright install chromium
    !python run_session.py --account colab1

Usage on Kaggle:
    Add to a Kaggle notebook cell and run.

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import os
import sys
import json
import time
import subprocess
import platform
import argparse
import socket


# ==============================================
# CONFIGURATION
# ==============================================

CONFIG_FILE = "config.json"
DEFAULT_GAMES = 10

# Account detection: map hostname patterns to account names
HOSTNAME_MAP = {
    # These are just defaults; user can always specify --account
}


# ==============================================
# SETUP FUNCTIONS
# ==============================================

def detect_platform() -> str:
    """Detect if running on Colab, Kaggle, or local."""
    if os.path.exists("/content"):
        return "colab"
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def detect_account(platform_name: str) -> str:
    """
    Try to auto-detect which account this is based on hostname.
    Falls back to asking the user.
    """
    hostname = socket.gethostname()

    # Check hostname map
    if hostname in HOSTNAME_MAP:
        return HOSTNAME_MAP[hostname]

    # Try environment variable
    account = os.environ.get("TERRITORIAL_ACCOUNT")
    if account:
        return account

    # Ask the user
    print("\n🔍 Could not auto-detect account name.")
    print("   Available accounts: colab1, colab2, colab3, colab4, kaggle1, kaggle2")

    while True:
        try:
            account = input("   Enter account name: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n   Using default: colab1")
            return "colab1"

        valid_accounts = ["colab1", "colab2", "colab3", "colab4", "kaggle1", "kaggle2"]
        if account in valid_accounts:
            return account
        print(f"   ⚠️ Invalid account. Choose from: {', '.join(valid_accounts)}")


def install_playwright():
    """Install Playwright and Chromium if not already installed."""
    try:
        import playwright
        print("   ✅ Playwright already installed")
    except ImportError:
        print("   📦 Installing Playwright...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])

    # Install Chromium browser
    print("   🌐 Ensuring Chromium is installed...")
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ Chromium ready")
    except subprocess.CalledProcessError:
        # On some platforms, need system deps first
        try:
            subprocess.check_call(
                [sys.executable, "-m", "playwright", "install-deps", "chromium"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.check_call(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("   ✅ Chromium ready (with deps)")
        except Exception as e:
            print(f"   ⚠️ Chromium install issue: {e}")
            print("   Continuing anyway — may work if already installed")


def load_config() -> dict:
    """Load configuration from config.json."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def pull_master_model():
    """Pull the latest master model from GitHub."""
    try:
        from sync import GitHubSync
        sync = GitHubSync()
        print("\n🔄 Pulling latest master model from GitHub...")
        success = sync.pull_master()
        if success:
            print("   ✅ Master model pulled successfully")
        else:
            print("   ⚠️ No master model on GitHub yet — starting fresh")
        return success
    except ImportError:
        print("   ⚠️ sync.py not available — skipping GitHub pull")
        return False
    except Exception as e:
        print(f"   ⚠️ Could not pull master model: {e}")
        return False


def load_master_weights(agent):
    """Load master model weights into the agent if available."""
    import torch

    master_model_json = "master_model.json"
    if os.path.exists(master_model_json):
        try:
            with open(master_model_json, 'r') as f:
                master_info = json.load(f)

            weights_path = master_info.get("model_weights_path", "models/master_model.pth")
            if os.path.exists(weights_path):
                agent.load_model(weights_path)
                print(f"   ✅ Loaded master model weights from {weights_path}")
                return True
            else:
                print(f"   ⚠️ Model weights file not found: {weights_path}")
        except Exception as e:
            print(f"   ⚠️ Could not load master weights: {e}")

    # Try account-specific model
    return False


def push_results(account_name: str):
    """Push session file and review queue to GitHub."""
    try:
        from sync import GitHubSync
        sync = GitHubSync()

        print("\n🔄 Syncing results to GitHub...")
        sync.push_session(account_name)
        print("   ✅ Session data pushed")

    except ImportError:
        print("   ⚠️ sync.py not available — skipping GitHub push")
    except Exception as e:
        print(f"   ⚠️ Could not push results: {e}")


def export_unrated_moves(recorder):
    """Export unrated moves to the review queue."""
    try:
        recorder.export_unrated()
    except Exception as e:
        print(f"   ⚠️ Could not export unrated moves: {e}")


# ==============================================
# MAIN SESSION RUNNER
# ==============================================

def run_session(account_name: str, num_games: int = DEFAULT_GAMES,
                use_real_game: bool = False):
    """
    Run a full training session.
    
    Args:
        account_name: e.g., "colab1", "kaggle2"
        num_games:    How many games to play
        use_real_game: If True, use TerritorialEnvironment (Playwright).
                       If False, use FakeGameEnvironment (testing).
    """
    start_time = time.time()

    print("\n" + "=" * 60)
    print(f"🚀 TERRITORIAL.IO AI — SESSION LAUNCHER")
    print(f"=" * 60)
    print(f"  Account:   {account_name}")
    print(f"  Games:     {num_games}")
    print(f"  Platform:  {detect_platform()}")
    print(f"  Real game: {use_real_game}")

    # --- Step 1: Install requirements ---
    if use_real_game:
        print("\n📦 Step 1: Installing requirements...")
        install_playwright()
    else:
        print("\n📦 Step 1: Using FakeGameEnvironment (no Playwright needed)")

    # --- Step 2: Pull latest master model ---
    print("\n📥 Step 2: Pulling latest master model...")
    pull_master_model()

    # --- Step 3: Load AI agent and master weights ---
    print("\n🧠 Step 3: Loading AI agent...")
    from brain import GameAgent
    agent = GameAgent()
    load_master_weights(agent)

    # --- Step 4: Run training session ---
    print(f"\n🎮 Step 4: Running {num_games} games...")

    from trainer import SessionTrainer
    trainer = SessionTrainer(account_name=account_name)

    # If we loaded master weights, apply them
    if os.path.exists("models/master_model.pth"):
        try:
            trainer.agent.load_model("models/master_model.pth")
        except Exception:
            pass  # Already tried loading above

    trainer.run_session(num_games=num_games)

    # --- Step 5: Export unrated moves ---
    print("\n📋 Step 5: Exporting unrated moves to review queue...")
    from move_recorder import MoveRecorder
    recorder = MoveRecorder(account_name)

    # Build session file
    session_filepath = f"session_{account_name}.json"
    session_data = trainer._compile_session_data()

    # Add extra fields for the session file format
    session_data["session_name"] = account_name
    session_data["platform"] = detect_platform()
    session_data["total_games_played"] = trainer.stats.games_played
    session_data["win_rate"] = trainer.stats.win_rate
    session_data["current_phase"] = trainer.curriculum.get_current_phase().value
    session_data["current_difficulty"] = trainer.curriculum.get_current_difficulty().value
    session_data["model_checkpoint_path"] = f"models/model_{account_name}.pth"
    session_data["best_strategies"] = []
    session_data["mistakes_to_avoid"] = []
    session_data["greenbiscuit_stats"] = {"following_opening": True}
    session_data["factor_averages"] = trainer.stats.get_factor_averages()
    session_data["last_updated"] = time.time()

    with open(session_filepath, 'w') as f:
        json.dump(session_data, f, indent=2, default=str)
    print(f"   💾 Session saved: {session_filepath}")

    # --- Step 6: Push results to GitHub ---
    print("\n🔄 Step 6: Pushing results to GitHub...")
    push_results(account_name)

    # --- Step 7: Print final summary ---
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 SESSION COMPLETE — {account_name}")
    print(f"{'='*60}")
    print(f"  ⏱️  Duration:        {elapsed / 60:.1f} minutes")
    print(f"  🎮 Games Played:    {trainer.stats.games_played}")
    print(f"  🏆 Wins:            {trainer.stats.games_won}")
    print(f"  💀 Losses:          {trainer.stats.games_lost}")
    print(f"  📈 Win Rate:        {trainer.stats.win_rate:.1%}")
    print(f"  🗺️  Best Territory:  {trainer.stats.best_territory:.1%}")
    worst_terr = min(trainer.stats.territory_per_game) if trainer.stats.territory_per_game else 0
    avg_terr = sum(trainer.stats.territory_per_game) / len(trainer.stats.territory_per_game) if trainer.stats.territory_per_game else 0
    print(f"  📉 Worst Territory: {worst_terr:.1%}")
    print(f"  📊 Avg Territory:   {avg_terr:.1%}")
    print(f"  📋 Moves needing review: {recorder.get_unrated_count()}")
    print(f"  🎓 Current Phase:   {trainer.curriculum.get_current_phase().value}")
    print(f"  ⚡ Current Difficulty: {trainer.curriculum.get_current_difficulty().value}")
    print(f"{'='*60}")


# ==============================================
# ENTRY POINT
# ==============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Territorial.io AI — Session Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_session.py --account colab1 --games 10
  python run_session.py --account kaggle1 --games 5 --real
  python run_session.py  # Auto-detects account
        """
    )

    parser.add_argument(
        "--account", "-a",
        type=str,
        default=None,
        help="Account name (colab1-4, kaggle1-2)"
    )
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=DEFAULT_GAMES,
        help=f"Number of games per session (default: {DEFAULT_GAMES})"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Playwright game environment (default: FakeGameEnvironment)"
    )

    args = parser.parse_args()

    # Detect account if not specified
    current_platform = detect_platform()
    account = args.account or detect_account(current_platform)

    run_session(
        account_name=account,
        num_games=args.games,
        use_real_game=args.real,
    )
