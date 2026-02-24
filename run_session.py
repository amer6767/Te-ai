"""
=============================================================================
run_session.py — Main Game Loop for Territorial.io LLM AI
=============================================================================

THE file to run on Kaggle/Colab. This is the Grand Finale that glues
everything together:

  1. Loads the LLM Commander (8B model in 4-bit)
  2. Launches Playwright browser → Territorial.io
  3. Runs an async game loop:
       Radar → LLM → Command → Click → Repeat

Usage on Kaggle:
    !pip install playwright transformers bitsandbytes accelerate
    !playwright install chromium
    !python run_session.py

Usage locally:
    python run_session.py --games 5 --headed

=============================================================================
"""

# ==============================================
# IMPORTS
# ==============================================

import os
import sys
import asyncio
import argparse
import time
import subprocess

from playwright.async_api import async_playwright

from llm_commander import LLMCommander
from game_environment import TerritorialEnvironment, get_state
from screen_capture import ScreenCapture


# ==============================================
# CONFIGURATION
# ==============================================

DEFAULT_GAMES = 3          # Games per session
MAX_STEPS_PER_GAME = 200   # Safety cap per game
LLM_CALL_INTERVAL = 3      # Only call LLM every N steps (save GPU time)


# ==============================================
# SETUP HELPERS
# ==============================================

def detect_platform() -> str:
    """Detect if running on Colab, Kaggle, or local."""
    if os.path.exists("/content"):
        return "colab"
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def install_playwright():
    """Install Playwright and Chromium if needed."""
    try:
        import playwright
        print("   ✅ Playwright already installed")
    except ImportError:
        print("   📦 Installing Playwright...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])

    print("   🌐 Ensuring Chromium is installed...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("   ✅ Chromium ready")
    except subprocess.CalledProcessError:
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


# ==============================================
# THE MAIN GAME LOOP
# ==============================================

async def play_one_game(
    env: TerritorialEnvironment,
    commander: LLMCommander,
    capture: ScreenCapture,
    game_num: int,
) -> dict:
    """
    Play a single game of Territorial.io with the LLM Commander.

    Flow per step:
      1. ScreenCapture → radar_text
      2. get_state() → troops, interest, etc.
      3. LLMCommander.decide(state, radar) → {"direction", "slider_pct"}
      4. env.step(command) → screenshot, reward, done, info

    The LLM is only called every LLM_CALL_INTERVAL steps to save GPU.
    Between calls, the last command is reused (the environment's own
    infinite-expansion handles ticks 2-4 anyway).

    Returns:
        dict with game stats
    """
    print(f"\n{'='*55}")
    print(f"🎮 GAME {game_num} — Starting...")
    print(f"{'='*55}")

    game_start = time.time()

    # Reset environment (navigate, spawn, zoom out)
    await env.reset()

    # Tell memory system: new game
    commander.start_new_game()

    # Re-attach capture to the new page
    capture.page = env.page

    # Track stats
    total_reward = 0.0
    best_territory = 0.0
    llm_calls = 0
    last_command = {"target_zone": 0, "slider_pct": 0}

    step = 0
    while step < MAX_STEPS_PER_GAME:
        step += 1

        # ----- EYES: Get grid scan -----
        try:
            frame = await capture.get_processed_frame()
            grid_text = frame["grid_text"]
        except Exception as e:
            print(f"   ⚠️ Capture error at step {step}: {e}")
            grid_text = "Zone 1: Unknown | Zone 2: Unknown | Zone 3: Unknown | Zone 4: Unknown | Zone 5: Our Territory | Zone 6: Unknown | Zone 7: Unknown | Zone 8: Unknown | Zone 9: Unknown"

        # ----- BRAIN: Ask LLM (every N steps) -----
        if step % LLM_CALL_INTERVAL == 1 or step == 1:
            try:
                game_state = await get_state(env.page)
                # Inject cycle/tick so the LLM sees them
                game_state["cycle"] = env.current_cycle
                game_state["tick"] = env.current_tick
                command = commander.decide(game_state, grid_text)
                last_command = command
                llm_calls += 1
            except Exception as e:
                print(f"   ⚠️ LLM error at step {step}: {e}")
                command = last_command
        else:
            command = last_command

        # ----- HANDS: Execute the command -----
        territory_before = env.prev_territory
        troops_before = env.last_troops
        try:
            screenshot, reward, done, info = await env.step(command)
        except Exception as e:
            print(f"   ❌ Step error: {e}")
            break

        # ----- REFLECT: Record outcome to memory -----
        territory_after = info.get("territory", 0.0)
        troops_after = info.get("troops", 0)
        action_str = info.get("action_taken", "unknown")
        try:
            commander.record_outcome(
                game_state=game_state if 'game_state' in dir() else {},
                action_str=action_str,
                territory_before=territory_before,
                territory_after=territory_after,
                troops_before=troops_before,
                troops_after=troops_after,
                grid_text=grid_text,
            )
        except Exception as e:
            pass  # Don't crash the game over a memory write failure

        total_reward += reward
        territory = info.get("territory", 0.0)
        best_territory = max(best_territory, territory)

        # Log every 10 steps
        if step % 10 == 0:
            print(
                f"   📊 Step {step:3d} | "
                f"C{info.get('cycle', '?')}:T{info.get('tick', '?')} | "
                f"Territory: {territory:.2%} | "
                f"Troops: {info.get('troops', 0):,} | "
                f"Action: {info.get('action_taken', '?')} | "
                f"Reward: {reward:+.2f}"
            )

        if done:
            won = info.get("won", False)
            reason = "🏆 VICTORY!" if won else "💀 Game Over"
            print(f"\n   {reason} at step {step}")
            break

    # ----- END OF GAME: Save memories -----
    final_territory = best_territory
    won = info.get("won", False) if 'info' in dir() else False
    try:
        commander.end_game(won=won, final_territory=final_territory * 100)
    except Exception as e:
        print(f"   ⚠️ Memory save error: {e}")

    game_time = time.time() - game_start

    stats = {
        "game": game_num,
        "steps": step,
        "total_reward": round(total_reward, 2),
        "best_territory": round(best_territory, 4),
        "llm_calls": llm_calls,
        "game_time_s": round(game_time, 1),
        "won": info.get("won", False) if 'info' in dir() else False,
    }

    print(f"\n   📈 Game {game_num} Summary:")
    print(f"      Steps: {stats['steps']} | Best Territory: {stats['best_territory']:.2%}")
    print(f"      LLM Calls: {stats['llm_calls']} | Time: {stats['game_time_s']:.0f}s")

    return stats


async def main(num_games: int = DEFAULT_GAMES, headless: bool = True):
    """
    The Grand Finale: load LLM, launch browser, play games.
    """
    session_start = time.time()

    print("\n" + "=" * 60)
    print("🚀 TERRITORIAL.IO — LLM COMMANDER SESSION")
    print("=" * 60)
    print(f"  Platform:  {detect_platform()}")
    print(f"  Games:     {num_games}")
    print(f"  Headless:  {headless}")
    print(f"  LLM Call Interval: every {LLM_CALL_INTERVAL} steps")

    # ---- Step 1: Load the LLM ----
    print("\n🧠 Step 1: Loading LLM Commander...")
    commander = LLMCommander()
    print(f"   ✅ LLM ready")

    # ---- Step 2: Launch Browser ----
    print("\n🌐 Step 2: Launching browser...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )

        env = TerritorialEnvironment(browser)

        # We create a temporary ScreenCapture — page will be reassigned after reset()
        capture = ScreenCapture(None)

        # ---- Step 3: Play Games ----
        print(f"\n🎮 Step 3: Playing {num_games} games...\n")
        all_stats = []

        for game_num in range(1, num_games + 1):
            try:
                stats = await play_one_game(env, commander, capture, game_num)
                all_stats.append(stats)
            except Exception as e:
                print(f"\n   ❌ Game {game_num} crashed: {e}")
                import traceback
                traceback.print_exc()

        # ---- Cleanup ----
        await env.close()
        await browser.close()

    # ---- Step 4: Session Summary ----
    session_time = time.time() - session_start

    print(f"\n{'='*60}")
    print(f"🏁 SESSION COMPLETE")
    print(f"{'='*60}")

    if all_stats:
        wins = sum(1 for s in all_stats if s.get("won"))
        avg_territory = sum(s["best_territory"] for s in all_stats) / len(all_stats)
        total_llm = sum(s["llm_calls"] for s in all_stats)

        print(f"  ⏱️  Duration:         {session_time / 60:.1f} minutes")
        print(f"  🎮 Games Played:     {len(all_stats)}")
        print(f"  🏆 Wins:             {wins}")
        print(f"  📊 Avg Best Territory: {avg_territory:.2%}")
        print(f"  🧠 Total LLM Calls:  {total_llm}")
        print(f"  📡 LLM Stats:        {commander.get_stats()}")
        print(f"  🧠 Memory Entries:   {commander.memory.get_stats()['long_term_entries']}")
    else:
        print("  No games completed.")

    print(f"{'='*60}\n")


# ==============================================
# ENTRY POINT
# ==============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Territorial.io — LLM Commander Session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_session.py                    # 3 games, headless
  python run_session.py --games 5          # 5 games
  python run_session.py --games 1 --headed # 1 game, visible browser
        """
    )

    parser.add_argument(
        "--games", "-g", type=int, default=DEFAULT_GAMES,
        help=f"Number of games per session (default: {DEFAULT_GAMES})"
    )
    parser.add_argument(
        "--headed", action="store_true",
        help="Run with visible browser window (default: headless)"
    )

    args = parser.parse_args()

    # Ensure Playwright is installed
    print("\n📦 Checking requirements...")
    install_playwright()

    asyncio.run(main(num_games=args.games, headless=not args.headed))
