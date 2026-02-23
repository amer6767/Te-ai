"""
================================================================================
🏋️ TRAINER.PY — SESSION RUNNER FOR TERRITORIAL.IO AI
================================================================================

This file runs actual training sessions:
  - Runs 10 game attempts per session
  - Each attempt plays a full game using brain.py and decision.py
  - After each move, records everything to memory.py
  - After each game, scores moves using rewards.py
  - Checks curriculum.py for current phase and difficulty
  - Saves everything to a session file: session_<account>.json

This is the "main" file — run this to train the AI!

HOW TO RUN:
  1. Upload all files to Google Colab
  2. Run: python trainer.py
  3. Or: python trainer.py --account colab1 --games 10

Requires: brain.py, decision.py, memory.py, rewards.py, curriculum.py

================================================================================
"""

import json
import os
import time
import random
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import asyncio
from playwright.async_api import async_playwright

import numpy as np
import torch

# ==============================================================================
# 📦 IMPORTS FROM OUR OTHER FILES
# ==============================================================================
# These import from the other files we built:

from brain import (
    Config,              # All settings (image size, learning rate, etc.)
    GamePreprocessor,    # Converts screenshots to tensors
    TerritorialAI,       # The CNN + Decision Network model
    GameAgent,           # Full agent with training logic
    FakeGameEnvironment  # Simulated game for testing
)

from decision import (
    ShortTermMemory,         # Remembers last N actions
    StrategicThinker,        # Dueling DQN neural network
    ExplorationStrategy,     # Epsilon-greedy / Boltzmann / UCB
    StrategicAdvisor,        # Recommends strategy type
    MasterDecisionEngine     # Combines everything
)

from memory import (
    MoveRecord,          # One move's data
    GameMemory,          # All moves in one game
    SessionMemory,       # Multiple games in one session
    FACTOR_NAMES         # The 10 rating factor names
)

from rewards import (
    MoveScorer,          # Scores all 10 factors
    win_loss_bonus       # Big bonus/penalty for game outcome
)

from curriculum import (
    CurriculumManager,   # Phase & difficulty management
    TrainingPhase,       # Phase 1 or Phase 2
    Difficulty,          # Easy, Medium, Hard
    UnratedMove          # Moves human missed rating
)

from game_environment import TerritorialEnvironment


# ==============================================================================
# 🌉 ASYNC BRIDGE FOR REAL GAME
# ==============================================================================

class RealGameBridge:
    """Wraps the async TerritorialEnvironment so the sync trainer can use it."""
    
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Start playwright and browser in the event loop
        self.playwright = self.loop.run_until_complete(async_playwright().start())
        self.browser = self.loop.run_until_complete(
            self.playwright.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        )
        self.env = TerritorialEnvironment(self.browser)
        
    def reset(self):
        return self.loop.run_until_complete(self.env.reset())
        
    def step(self, action):
        return self.loop.run_until_complete(self.env.step(action))
        
    def close(self):
        self.loop.run_until_complete(self.env.close())
        self.loop.run_until_complete(self.playwright.stop())
        self.loop.close()


# ==============================================================================
# ⚙️ SESSION CONFIGURATION
# ==============================================================================

class SessionConfig:
    """
    Configuration for one training session.
    A session = multiple games played back-to-back.
    """
    
    # --- Session Settings ---
    GAMES_PER_SESSION = 10           # Play 10 games per session
    MAX_MOVES_PER_GAME = 200         # Cap at 200 moves per game (safety)
    
    # --- Saving ---
    SESSION_DIR = "sessions"          # Folder to save session files
    MODEL_DIR = "models"              # Folder to save model checkpoints
    
    # --- Logging ---
    VERBOSE = True                    # Print detailed logs
    SAVE_SCREENSHOTS = False          # Save screenshots (uses disk space)
    SCREENSHOT_DIR = "screenshots"    # Where to save screenshots
    
    # --- Training ---
    TRAIN_AFTER_EVERY_MOVE = True     # Train the model after each move
    TRAIN_AFTER_GAME = True           # Extra training batch after each game
    POST_GAME_TRAIN_STEPS = 5         # How many extra training steps after game
    
    # --- Model Checkpoints ---
    SAVE_MODEL_EVERY_N_GAMES = 5      # Save model every 5 games
    KEEP_BEST_MODEL = True            # Keep a copy of the best-performing model


# ==============================================================================
# 📊 SESSION STATS — Track everything in real-time
# ==============================================================================

class SessionStats:
    """
    Tracks statistics across all games in the current session.
    
    Updated in real-time as games are played.
    Printed at the end for a summary.
    """
    
    def __init__(self, account_name: str):
        self.account_name = account_name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        
        # --- Game results ---
        self.games_played = 0
        self.games_won = 0
        self.games_lost = 0
        self.game_results: List[Dict] = []    # Detailed per-game results
        
        # --- Move stats ---
        self.total_moves = 0
        self.total_random_moves = 0           # Exploration moves
        self.total_smart_moves = 0            # Exploitation moves
        
        # --- Training stats ---
        self.total_train_steps = 0
        self.total_loss = 0.0
        self.loss_history: List[float] = []
        
        # --- Reward stats ---
        self.total_reward = 0.0
        self.best_game_reward = float('-inf')
        self.worst_game_reward = float('inf')
        self.reward_per_game: List[float] = []
        
        # --- Territory stats ---
        self.best_territory = 0.0
        self.avg_final_territory = 0.0
        self.territory_per_game: List[float] = []
        
        # --- Factor scores (the 10 factors) ---
        self.factor_totals: Dict[str, float] = {name: 0.0 for name in FACTOR_NAMES}
        self.factor_counts: Dict[str, int] = {name: 0 for name in FACTOR_NAMES}
    
    def record_game(self, result: Dict):
        """Record one game's results."""
        self.games_played += 1
        
        if result.get("won", False):
            self.games_won += 1
        else:
            self.games_lost += 1
        
        self.total_moves += result.get("moves", 0)
        self.total_reward += result.get("total_reward", 0)
        self.reward_per_game.append(result.get("total_reward", 0))
        
        territory = result.get("final_territory", 0)
        self.territory_per_game.append(territory)
        self.best_territory = max(self.best_territory, territory)
        
        if result.get("total_reward", 0) > self.best_game_reward:
            self.best_game_reward = result["total_reward"]
        if result.get("total_reward", 0) < self.worst_game_reward:
            self.worst_game_reward = result["total_reward"]
        
        self.game_results.append(result)
    
    def record_factor_score(self, factor_name: str, score: float):
        """Record a single factor score for averaging later."""
        if factor_name in self.factor_totals:
            self.factor_totals[factor_name] += score
            self.factor_counts[factor_name] += 1
    
    @property
    def win_rate(self) -> float:
        """Current session win rate."""
        if self.games_played == 0:
            return 0.0
        return self.games_won / self.games_played
    
    @property
    def avg_reward(self) -> float:
        """Average reward per game."""
        if self.games_played == 0:
            return 0.0
        return self.total_reward / self.games_played
    
    @property
    def duration_minutes(self) -> float:
        """How long the session has been running (minutes)."""
        end = self.end_time or time.time()
        return (end - self.start_time) / 60
    
    def get_factor_averages(self) -> Dict[str, float]:
        """Get average score for each of the 10 factors."""
        averages = {}
        for name in FACTOR_NAMES:
            if self.factor_counts[name] > 0:
                averages[name] = self.factor_totals[name] / self.factor_counts[name]
            else:
                averages[name] = 0.0
        return averages
    
    def print_summary(self):
        """Print a beautiful session summary."""
        self.end_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"📊 SESSION SUMMARY — {self.account_name}")
        print(f"{'='*70}")
        print(f"")
        print(f"  ⏱️  Duration:        {self.duration_minutes:.1f} minutes")
        print(f"  🎮 Games Played:    {self.games_played}")
        print(f"  🏆 Wins:            {self.games_won}")
        print(f"  💀 Losses:          {self.games_lost}")
        print(f"  📈 Win Rate:        {self.win_rate:.1%}")
        print(f"")
        print(f"  🎯 Total Moves:     {self.total_moves}")
        print(f"  🎲 Random Moves:    {self.total_random_moves} ({self.total_random_moves/(self.total_moves or 1):.0%})")
        print(f"  🧠 Smart Moves:     {self.total_smart_moves} ({self.total_smart_moves/(self.total_moves or 1):.0%})")
        print(f"")
        print(f"  💰 Total Reward:    {self.total_reward:.1f}")
        print(f"  📊 Avg Reward:      {self.avg_reward:.1f}")
        print(f"  ⭐ Best Game:       {self.best_game_reward:.1f}")
        print(f"  💩 Worst Game:      {self.worst_game_reward:.1f}")
        print(f"")
        print(f"  🗺️  Best Territory:  {self.best_territory:.1%}")
        print(f"  📍 Avg Territory:   {np.mean(self.territory_per_game):.1%}" if self.territory_per_game else "")
        print(f"")
        
        # Print factor averages
        factor_avgs = self.get_factor_averages()
        print(f"  📋 Average Factor Scores:")
        for name, avg in factor_avgs.items():
            bar_length = int(abs(avg) * 20)
            bar = "█" * bar_length
            sign = "+" if avg >= 0 else "-"
            print(f"     {name:30s} {sign}{abs(avg):.2f} {bar}")
        
        print(f"\n{'='*70}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving to JSON."""
        return {
            "account_name": self.account_name,
            "start_time": self.start_time,
            "end_time": self.end_time or time.time(),
            "duration_minutes": self.duration_minutes,
            "games_played": self.games_played,
            "games_won": self.games_won,
            "games_lost": self.games_lost,
            "win_rate": self.win_rate,
            "total_moves": self.total_moves,
            "total_random_moves": self.total_random_moves,
            "total_smart_moves": self.total_smart_moves,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "best_game_reward": self.best_game_reward,
            "worst_game_reward": self.worst_game_reward,
            "best_territory": self.best_territory,
            "reward_per_game": self.reward_per_game,
            "territory_per_game": self.territory_per_game,
            "factor_averages": self.get_factor_averages(),
            "game_results": self.game_results,
        }


# ==============================================================================
# 🎮 GAME RUNNER — Plays one full game
# ==============================================================================

class GameRunner:
    """
    Plays ONE full game from start to finish.
    
    Steps:
      1. Reset the environment
      2. Loop: take screenshot → decide action → execute → record → train
      3. When game ends: score all moves, return results
    
    Uses:
      - brain.py's GameAgent for the neural network
      - decision.py's MasterDecisionEngine for smart decisions
      - memory.py's GameMemory to record everything
      - rewards.py's MoveScorer to score the 10 factors
      - curriculum.py's training config for current settings
    """
    
    def __init__(
        self,
        agent: GameAgent,
        curriculum: CurriculumManager,
        game_id: int,
        use_real_game: bool = False,
        real_env = None,
        verbose: bool = True
    ):
        """
        Initialize a game runner.
        
        Args:
            agent:      The trained GameAgent from brain.py
            curriculum: The curriculum manager for settings
            game_id:    Unique ID for this game
            use_real_game: Whether to use the real playwright env
            real_env:   The RealGameBridge instance
            verbose:    Print move-by-move updates
        """
        self.agent = agent
        self.curriculum = curriculum
        self.game_id = game_id
        self.use_real_game = use_real_game
        self.real_env = real_env
        self.verbose = verbose
        
        # --- Get training config for current phase/difficulty ---
        self.config = curriculum.get_training_config()
        
        # --- Create game-specific components ---
        self.game_memory = GameMemory(game_id=game_id)
        self.move_scorer = MoveScorer()
        self.preprocessor = GamePreprocessor()
        
        # --- Create environment based on difficulty ---
        self.env = self._create_environment()
        
        # --- Track game state ---
        self.move_count = 0
        self.total_reward = 0.0
        self.unrated_moves: List[UnratedMove] = []
    
    def _create_environment(self):
        """
        Create a game environment matching the current difficulty.
        
        In a real setup, this would connect to the actual game.
        For now, it uses the FakeGameEnvironment with different settings.
        """
        if self.use_real_game and self.real_env:
            return self.real_env

        difficulty = self.config["difficulty"]
        
        if difficulty == "easy":
            env = FakeGameEnvironment(width=300, height=300)
            env.max_steps = 60         # Short games
        elif difficulty == "medium":
            env = FakeGameEnvironment(width=400, height=400)
            env.max_steps = 100        # Medium games
        else:  # hard
            env = FakeGameEnvironment(width=500, height=500)
            env.max_steps = 150        # Long games
        
        return env
    
    def play_full_game(self) -> Dict:
        """
        Play one complete game from start to finish.
        
        Returns:
            Dictionary with game results including:
              - won: bool
              - moves: int
              - total_reward: float
              - final_territory: float
              - move_scores: list of factor scores
        """
        # --- Step 1: Reset environment, get first screenshot ---
        screenshot = self.env.reset()
        prev_screenshot = screenshot
        state_tensor = self.preprocessor.process_screenshot(screenshot)
        
        if self.verbose:
            print(f"\n  🎮 Game {self.game_id} starting... "
                  f"[{self.config['difficulty'].upper()}]")
        
        # --- Step 2: Game loop — keep playing until game ends ---
        game_over = False
        won = False
        final_territory = 0.0
        
        while not game_over and self.move_count < SessionConfig.MAX_MOVES_PER_GAME:
            # --- 2a: Choose an action ---
            action = self.agent.select_action(screenshot)
            self.move_count += 1
            
            # --- 2b: Execute the action in the game ---
            next_screenshot, reward, done, info = self.env.step(action)
            
            # --- 2c: Calculate the 10 factor scores for this move ---
            #     In Phase 1, some factors need human rating
            #     In Phase 2, we auto-score everything
            factor_scores = self._score_move(
                prev_screenshot=prev_screenshot,
                curr_screenshot=next_screenshot,
                action=action,
                reward=reward,
                info=info
            )
            
            # --- 2d: Record the move in memory ---
            # BUG 1 FIX: MoveRecord uses (move_number, action, screenshot, metadata)
            metadata = {
                "reward": reward,
                "factor_scores": factor_scores,
                "was_random": action.get("was_random", False),
                "timestamp": time.time(),
                "info": info,
            }
            screenshot_array = np.array(screenshot) if hasattr(screenshot, '__array__') else None
            self.game_memory.add_move(
                action=action,
                screenshot=screenshot_array,
                metadata=metadata
            )
            
            # --- 2e: Store experience for neural network training ---
            next_state_tensor = self.preprocessor.process_screenshot(next_screenshot)
            self.agent.store_experience(
                state_tensor, 
                action["action_index"], 
                reward, 
                next_state_tensor, 
                done
            )
            
            # --- 2f: Train the neural network (if configured) ---
            if SessionConfig.TRAIN_AFTER_EVERY_MOVE:
                loss = self.agent.train_step()
            
            # --- 2g: Update tracking variables ---
            self.total_reward += reward
            prev_screenshot = screenshot
            screenshot = next_screenshot
            state_tensor = next_state_tensor
            final_territory = info.get("territory", 0)
            
            # --- 2h: Check if game is over ---
            if done:
                game_over = True
                won = info.get("won", False)
            
            # --- 2i: Print move update (every 20 moves) ---
            if self.verbose and self.move_count % 20 == 0:
                print(f"    Move {self.move_count:3d} | "
                      f"Territory: {final_territory:.1%} | "
                      f"Reward: {self.total_reward:.1f} | "
                      f"{'🎲' if action['was_random'] else '🧠'} {action['action_type']}")
        
        # --- Step 3: Game over — add win/loss bonus ---
        outcome_bonus = win_loss_bonus(won)
        self.total_reward += outcome_bonus
        
        # --- Step 4: Post-game extra training ---
        if SessionConfig.TRAIN_AFTER_GAME:
            for _ in range(SessionConfig.POST_GAME_TRAIN_STEPS):
                self.agent.train_step()
        
        # --- Step 5: Compile results ---
        result = {
            "game_id": self.game_id,
            "won": won,
            "moves": self.move_count,
            "total_reward": self.total_reward,
            "final_territory": final_territory,
            "difficulty": self.config["difficulty"],
            "phase": self.config["phase"],
            "timestamp": time.time(),
        }
        
        # --- Step 6: Print result ---
        if self.verbose:
            emoji = "🏆 WIN!" if won else "💀 LOSS"
            print(f"  {emoji} Game {self.game_id} | "
                  f"{self.move_count} moves | "
                  f"Territory: {final_territory:.1%} | "
                  f"Reward: {self.total_reward:.1f}")
        
        return result
    
    def _score_move(
        self,
        prev_screenshot,
        curr_screenshot,
        action: Dict,
        reward: float,
        info: Dict
    ) -> Dict[str, Optional[float]]:
        """
        Score a single move across all 10 factors.
        
        In Phase 1: Some factors auto-scored, others left as None for human
        In Phase 2: All factors auto-scored
        
        Returns:
            Dictionary of {factor_name: score} where score is -1.0 to 1.0
        """
        # Get current game state for scoring
        territory = info.get("territory", 0.0)
        step = info.get("step", 0)
        num_players = info.get("num_players", self.config.get("num_opponents", 5) + 1)
        
        # Estimate previous territory from territory change
        prev_territory = max(0, territory - 0.02)  # REPLACE WITH REAL DATA FROM screen_capture.py
        
        # BUG 2 FIX: Use score_move() instead of score_all_factors()
        scores = self.move_scorer.score_move(
            action=action,
            prev_territory_pct=prev_territory,
            curr_territory_pct=territory,
            move_number=self.move_count,
            territory_pct=territory,
            num_players=num_players,
            is_enemy_fighting=random.random() < 0.3,          # REPLACE WITH REAL DATA FROM screen_capture.py
            target_is_weakest=None,                           # REPLACE WITH REAL DATA FROM screen_capture.py
            target_is_neighbor=None,                          # REPLACE WITH REAL DATA FROM screen_capture.py
            our_border_tiles=random.randint(10, 100),         # REPLACE WITH REAL DATA FROM screen_capture.py
            our_total_tiles=int(territory * 1000),            # REPLACE WITH REAL DATA FROM screen_capture.py
            our_power=random.uniform(0.3, 1.0),               # REPLACE WITH REAL DATA FROM screen_capture.py
            enemy_power=random.uniform(0.2, 0.8),             # REPLACE WITH REAL DATA FROM screen_capture.py
            power_before=random.uniform(0.3, 1.0),            # REPLACE WITH REAL DATA FROM screen_capture.py
            power_after=random.uniform(0.1, 0.8),             # REPLACE WITH REAL DATA FROM screen_capture.py
            power_max=1.0,                                    # REPLACE WITH REAL DATA FROM screen_capture.py
            undefended_nearby=random.random() < 0.2,          # REPLACE WITH REAL DATA FROM screen_capture.py
            enemy_just_lost=random.random() < 0.1,            # REPLACE WITH REAL DATA FROM screen_capture.py
            weak_enemy_nearby=random.random() < 0.2,          # REPLACE WITH REAL DATA FROM screen_capture.py
            got_backstabbed=False,                             # REPLACE WITH REAL DATA FROM screen_capture.py
            territory_center=None,                             # REPLACE WITH REAL DATA FROM screen_capture.py
            has_corner=random.random() < 0.3,                 # REPLACE WITH REAL DATA FROM screen_capture.py
            has_edge=random.random() < 0.5,                   # REPLACE WITH REAL DATA FROM screen_capture.py
            game_over=info.get("game_over", False),
            we_died=info.get("game_over", False) and not info.get("won", False),
        )
        
        return scores
    
    def get_game_memory(self) -> GameMemory:
        """Get the complete game memory with all recorded moves."""
        return self.game_memory
    
    def get_unrated_moves(self) -> List[UnratedMove]:
        """Get any moves that need human rating (Phase 1)."""
        return self.unrated_moves


# ==============================================================================
# 🏋️ SESSION TRAINER — Runs multiple games
# ==============================================================================

class SessionTrainer:
    """
    The main trainer that runs a full training session.
    
    A session consists of:
      1. Loading the AI model (or creating a new one)
      2. Checking curriculum for current phase/difficulty
      3. Playing N games (default 10)
      4. Recording everything to memory
      5. Scoring all moves
      6. Saving everything to a session file
    
    Usage:
        trainer = SessionTrainer(account_name="colab1")
        trainer.run_session(num_games=10)
    """
    
    def __init__(self, account_name: str = "default", use_real_game: bool = False):
        """
        Initialize the session trainer.
        
        Args:
            account_name: Name for this training account
                         Used in save file: session_<account>.json
            use_real_game: Whether to use the real Playwright environment
        """
        self.account_name = account_name
        self.stats = SessionStats(account_name)
        self.use_real_game = use_real_game
        
        self.real_env = None
        if self.use_real_game:
            print("\n🌐 Initializing Playwright browser (RealGameBridge)...")
            self.real_env = RealGameBridge()
        
        # --- Create directories ---
        os.makedirs(SessionConfig.SESSION_DIR, exist_ok=True)
        os.makedirs(SessionConfig.MODEL_DIR, exist_ok=True)
        
        # --- Initialize components ---
        print(f"\n{'='*60}")
        print(f"🏋️ TERRITORIAL.IO TRAINER — Account: {account_name}")
        print(f"{'='*60}")
        
        # Load or create the AI agent
        print("\n📦 Loading AI components...")
        self.agent = GameAgent()
        self._load_model_if_exists()
        
        # Load curriculum state
        print("📚 Loading curriculum...")
        curriculum_path = os.path.join(
            SessionConfig.SESSION_DIR, 
            f"curriculum_{account_name}.json"
        )
        self.curriculum = CurriculumManager(save_path=curriculum_path)
        
        # Create session memory
        # BUG 3 FIX: SessionMemory uses session_name, not session_id
        self.session_memory = SessionMemory(session_name=account_name)
        
        # Print current state
        print(self.curriculum.get_phase_description())
    
    def _load_model_if_exists(self):
        """Load saved model if one exists for this account."""
        model_path = os.path.join(
            SessionConfig.MODEL_DIR, 
            f"model_{self.account_name}.pth"
        )
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            print(f"   ✅ Loaded existing model from {model_path}")
        else:
            print(f"   🆕 No saved model found — starting fresh!")
    
    def _save_model(self, suffix: str = ""):
        """Save the current model."""
        filename = f"model_{self.account_name}{suffix}.pth"
        model_path = os.path.join(SessionConfig.MODEL_DIR, filename)
        self.agent.save_model(model_path)
    
    def run_session(self, num_games: Optional[int] = None):
        """
        Run a full training session with multiple games.
        
        This is the main function! Call this to start training.
        
        Args:
            num_games: How many games to play (default: 10)
        """
        num_games = num_games or SessionConfig.GAMES_PER_SESSION
        
        print(f"\n{'🎮'*30}")
        print(f"🎮 STARTING SESSION: {num_games} games")
        print(f"   Phase: {self.curriculum.get_current_phase().value}")
        print(f"   Difficulty: {self.curriculum.get_current_difficulty().value}")
        print(f"{'🎮'*30}\n")
        
        # --- Play each game ---
        for game_num in range(1, num_games + 1):
            print(f"\n{'─'*50}")
            print(f"📍 Game {game_num} of {num_games}")
            print(f"{'─'*50}")
            
            try:
                # --- Create a game runner ---
                game_id = self.curriculum.total_games_played + 1
                runner = GameRunner(
                    agent=self.agent,
                    curriculum=self.curriculum,
                    game_id=game_id,
                    use_real_game=self.use_real_game,
                    real_env=self.real_env,
                    verbose=SessionConfig.VERBOSE
                )
                
                # --- Play the full game ---
                result = runner.play_full_game()
                
                # --- Record in session stats ---
                self.stats.record_game(result)
                
                # --- Record in curriculum ---
                self.curriculum.record_game_result(
                    won=result["won"],
                    moves_made=result["moves"],
                    territory_gained=result["final_territory"],
                    unrated_moves=runner.get_unrated_moves()
                )
                
                # --- Record in session memory ---
                # BUG 4 FIX: Use new_game() at start and end_current_game() at end
                # instead of add_game(). The new_game() was called implicitly;
                # here we just need to end the current game properly.
                self.session_memory.new_game()
                # Copy moves from runner's game memory to session memory
                game_mem = runner.get_game_memory()
                for move in game_mem:
                    self.session_memory.current_game.moves.append(move)
                    self.session_memory.current_game.move_count += 1
                self.session_memory.end_current_game(
                    won=result["won"],
                    final_territory=result["final_territory"]
                )
                
                # --- Update target network periodically ---
                if game_num % Config.TARGET_UPDATE == 0:
                    self.agent.update_target_network()
                    print("   🔄 Target network updated")
                
                # --- Save model checkpoint ---
                if game_num % SessionConfig.SAVE_MODEL_EVERY_N_GAMES == 0:
                    self._save_model()
                    print(f"   💾 Model checkpoint saved")
                
            except Exception as e:
                print(f"   ❌ Error in game {game_num}: {e}")
                traceback.print_exc()
                continue
        
        # --- Session complete ---
        self._finish_session()
    
    def _finish_session(self):
        """
        Finish the session: save everything, print summary.
        Called automatically after all games are played.
        """
        if self.real_env:
            print("\n🌐 Closing Playwright browser...")
            try:
                self.real_env.close()
            except Exception as e:
                print(f"⚠️ Error closing browser: {e}")

        print(f"\n{'='*60}")
        print(f"✅ SESSION COMPLETE!")
        print(f"{'='*60}")
        
        # --- Save final model ---
        self._save_model()
        
        # --- Save best model if this is the best session ---
        if SessionConfig.KEEP_BEST_MODEL:
            best_path = os.path.join(
                SessionConfig.MODEL_DIR, 
                f"model_{self.account_name}_best.pth"
            )
            # Save as best if win rate > 50% (simple heuristic)
            if self.stats.win_rate > 0.5:
                self._save_model(suffix="_best")
                print(f"   ⭐ New best model saved! (Win rate: {self.stats.win_rate:.0%})")
        
        # --- Save session data ---
        session_data = self._compile_session_data()
        session_filename = (
            f"session_{self.account_name}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        session_path = os.path.join(SessionConfig.SESSION_DIR, session_filename)
        
        try:
            with open(session_path, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            print(f"   📄 Session saved to: {session_path}")
        except Exception as e:
            print(f"   ⚠️ Could not save session: {e}")
        
        # --- Print summary ---
        self.stats.print_summary()
        
        # --- Print promotion progress ---
        progress = self.curriculum.get_promotion_progress()
        if isinstance(progress, dict) and "overall_progress" in progress:
            print(f"\n📈 Promotion Progress: {progress['overall_progress']:.0%}")
            print(f"   Games: {progress['games_played']['current']}/{progress['games_played']['needed']}")
            print(f"   Win Rate: {progress['win_rate']['current']:.1%} (need {progress['win_rate']['needed']:.0%})")
            print(f"   Best Streak: {progress['win_streak']['current']} (need {progress['win_streak']['needed']})")
    
    def _compile_session_data(self) -> Dict:
        """
        Compile all session data into one dictionary for saving.
        This creates the session_<account>.json file.
        """
        return {
            "metadata": {
                "account_name": self.account_name,
                "session_date": datetime.now().isoformat(),
                "duration_minutes": self.stats.duration_minutes,
                "phase": self.curriculum.get_current_phase().value,
                "difficulty": self.curriculum.get_current_difficulty().value,
                "total_games_ever": self.curriculum.total_games_played,
            },
            "session_stats": self.stats.to_dict(),
            "curriculum_state": {
                "phase": self.curriculum.get_current_phase().value,
                "difficulty": self.curriculum.get_current_difficulty().value,
                "total_games": self.curriculum.total_games_played,
                "promotion_progress": self.curriculum.get_promotion_progress(),
            },
            "model_info": {
                "epsilon": self.agent.epsilon,
                "steps_done": self.agent.steps_done,
                "episodes_done": self.agent.episodes_done,
                "memory_size": len(self.agent.memory),
            },
        }


# ==============================================================================
# 🚀 QUICK START — Run training with one command
# ==============================================================================

def quick_train(account: str = "colab1", games: int = 10):
    """
    Quick-start function to begin training immediately.
    
    Args:
        account: Name for this training account
        games:   How many games to play
    
    Usage:
        quick_train("colab1", 10)
    """
    trainer = SessionTrainer(account_name=account)
    trainer.run_session(num_games=games)
    return trainer


# ==============================================================================
# 🧪 DEMO MODE — Test everything works
# ==============================================================================

def demo():
    """
    Run a quick demo with 3 games to test everything works.
    Uses smaller settings so it runs fast.
    """
    print("\n🧪 DEMO MODE — Quick test with 3 games\n")
    
    # Use shorter games for demo
    SessionConfig.GAMES_PER_SESSION = 3
    SessionConfig.SAVE_MODEL_EVERY_N_GAMES = 3
    SessionConfig.VERBOSE = True
    
    trainer = SessionTrainer(account_name="demo")
    trainer.run_session(num_games=3)
    
    # Clean up demo files
    demo_files = [
        "sessions/curriculum_demo.json",
        "models/model_demo.pth",
    ]
    for f in demo_files:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n✅ Demo complete! Everything works.\n")


# ==============================================================================
# 🎯 MAIN — Entry point
# ==============================================================================

if __name__ == "__main__":
    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser(
        description="🏋️ Territorial.io AI Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trainer.py                          # Default: 10 games, account "colab1"
  python trainer.py --account mypc --games 20  # 20 games, account "mypc"  
  python trainer.py --demo                     # Quick 3-game test
        """
    )
    
    parser.add_argument(
        "--account", "-a",
        type=str,
        default="colab1",
        help="Account name for saving (default: colab1)"
    )
    parser.add_argument(
        "--games", "-g",
        type=int,
        default=10,
        help="Number of games per session (default: 10)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick 3-game demo"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Less verbose output"
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        SessionConfig.VERBOSE = False
    
    if args.demo:
        demo()
    else:
        print(f"\n🏋️ Starting training session...")
        print(f"   Account: {args.account}")
        print(f"   Games:   {args.games}")
        print(f"   Device:  {Config.DEVICE}")
        print()
        
        quick_train(account=args.account, games=args.games)
        
        print(f"\n🎉 Done! Session saved as session_{args.account}_*.json")
        print(f"   Model saved as models/model_{args.account}.pth")
