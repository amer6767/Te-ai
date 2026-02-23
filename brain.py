"""
=============================================================================
brain.py — Territorial.io AI Agent
=============================================================================
A CNN-based AI that plays Territorial.io by:
  1. Taking a screenshot of the game as input
  2. Extracting visual features (territory colors, borders, etc.)
  3. Deciding where to attack or move

Uses PyTorch — lightweight enough for free Google Colab / Kaggle GPUs.

HOW TO RUN:
  1. Open Google Colab (colab.research.google.com)
  2. Go to Runtime > Change runtime type > GPU
  3. Upload this file or paste the code
  4. Run all cells!

=============================================================================
"""

# =============================================
# STEP 1: IMPORTS
# =============================================
# These are the libraries we need

import torch                          # PyTorch — the AI framework
import torch.nn as nn                 # Neural network building blocks
import torch.nn.functional as F       # Activation functions (ReLU, softmax)
import torch.optim as optim           # Optimizers (Adam) to train the model
import numpy as np                    # Number crunching
from PIL import Image                 # For loading screenshots
import torchvision.transforms as T    # To preprocess images for the CNN
import random                         # For random exploration
from collections import deque         # Memory buffer for training
import os                             # File system stuff


# =============================================
# STEP 2: CONFIGURATION
# =============================================
# All the settings in one place — easy to tweak!

class Config:
    """
    All hyperparameters and settings for the AI.
    Change these to experiment!
    """

    # --- Image Settings ---
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    IMAGE_CHANNELS = 3            # RGB (Red, Green, Blue)

    # --- Game Grid ---
    # 16x16 grid = 256 possible positions to click/attack
    GRID_ROWS = 16
    GRID_COLS = 16

    # --- Action Space ---
    NUM_GRID_ACTIONS = GRID_ROWS * GRID_COLS   # 256 positions
    NUM_SPECIAL_ACTIONS = 3                      # 0=wait, 1=defend, 2=expand
    TOTAL_ACTIONS = NUM_GRID_ACTIONS + NUM_SPECIAL_ACTIONS  # 259 total

    # --- Training Settings ---
    LEARNING_RATE = 0.0003
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995
    BATCH_SIZE = 32
    MEMORY_SIZE = 10000
    TARGET_UPDATE = 10

    # --- Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Print what device we're using
print(f"🖥️  Using device: {Config.DEVICE}")
if Config.DEVICE.type == "cuda":
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU found — using CPU (will be slower)")


# =============================================
# STEP 3: IMAGE PREPROCESSING
# =============================================

class GamePreprocessor:
    """
    Takes a raw game screenshot and prepares it for the neural network.
    Resizes, normalizes, and converts to tensor.
    """

    def __init__(self):
        self.transform = T.Compose([
            T.Resize((Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_screenshot(self, image):
        """Convert a game screenshot to a tensor the CNN can understand."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"), "RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(Config.DEVICE)
        return tensor

    def extract_territory_colors(self, image):
        """Analyze the screenshot to find unique territory colors."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)

        small = np.array(Image.fromarray(image).resize((64, 64)))
        quantized = (small // 10) * 10
        pixels = quantized.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

        sorted_indices = np.argsort(-counts)
        unique_colors = unique_colors[sorted_indices]
        counts = counts[sorted_indices]

        total_pixels = len(pixels)
        significant = counts > (total_pixels * 0.01)

        return {
            "colors": unique_colors[significant],
            "counts": counts[significant],
            "percentages": counts[significant] / total_pixels * 100,
            "num_players": int(significant.sum()),
        }


# =============================================
# STEP 4: THE CNN — VISUAL FEATURE EXTRACTOR
# =============================================
# The "eyes" of the AI

class VisionCNN(nn.Module):
    """
    4-layer CNN that processes game screenshots.
    Input: [3, 128, 128] RGB image
    Output: [128] feature vector
    """

    def __init__(self):
        super(VisionCNN, self).__init__()

        # Layer 1: Color detector — 3→32 filters
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        # Layer 2: Shape detector — 32→64 filters
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        # Layer 3: Border detector — 64→128 filters
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        # Layer 4: Strategy detector — 128→128 filters
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x


# =============================================
# STEP 5: DECISION NETWORK
# =============================================

class DecisionNetwork(nn.Module):
    """
    Takes CNN features [128] and outputs Q-values [259].
    259 = 256 grid positions + wait + defend + expand
    """

    def __init__(self):
        super(DecisionNetwork, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, Config.TOTAL_ACTIONS)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# =============================================
# STEP 6: COMPLETE AI MODEL
# =============================================

class TerritorialAI(nn.Module):
    """CNN + Decision Network combined."""

    def __init__(self):
        super(TerritorialAI, self).__init__()
        self.vision = VisionCNN()
        self.decision = DecisionNetwork()
        self.to(Config.DEVICE)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n🧠 TerritorialAI Model Created!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    def forward(self, screenshot_tensor):
        features = self.vision(screenshot_tensor)
        q_values = self.decision(features)
        return q_values


# =============================================
# STEP 7: GAME AGENT
# =============================================

class GameAgent:
    """
    Full agent with epsilon-greedy exploration,
    experience replay, and training loop.
    """

    def __init__(self):
        self.policy_net = TerritorialAI()
        self.target_net = TerritorialAI()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.preprocessor = GamePreprocessor()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.epsilon = Config.EPSILON_START
        self.steps_done = 0
        self.episodes_done = 0
        self.total_rewards = []

        print(f"\n🎮 Game Agent Ready!")

    def select_action(self, screenshot):
        """Pick an action: random (explore) or smart (exploit)."""
        self.steps_done += 1

        if random.random() < self.epsilon:
            action_idx = random.randint(0, Config.TOTAL_ACTIONS - 1)
            was_random = True
            confidence = 0.0
        else:
            state_tensor = self.preprocessor.process_screenshot(screenshot)
            with torch.no_grad():
                self.policy_net.eval()
                q_values = self.policy_net(state_tensor)
                self.policy_net.train()
            action_idx = q_values.argmax(dim=1).item()
            was_random = False
            q_vals = q_values.squeeze()
            confidence = float(torch.softmax(q_vals, dim=0).max().item())

        return self._decode_action(action_idx, was_random, confidence)

    def _decode_action(self, action_idx, was_random, confidence):
        result = {
            "action_index": action_idx, "was_random": was_random,
            "confidence": confidence, "grid_row": None, "grid_col": None,
            "screen_x": None, "screen_y": None,
        }
        if action_idx < Config.NUM_GRID_ACTIONS:
            row = action_idx // Config.GRID_COLS
            col = action_idx % Config.GRID_COLS
            result["action_type"] = "click"
            result["grid_row"] = row
            result["grid_col"] = col
            result["screen_x"] = (col + 0.5) / Config.GRID_COLS
            result["screen_y"] = (row + 0.5) / Config.GRID_ROWS
        elif action_idx == Config.NUM_GRID_ACTIONS:
            result["action_type"] = "wait"
        elif action_idx == Config.NUM_GRID_ACTIONS + 1:
            result["action_type"] = "defend"
        else:
            result["action_type"] = "expand"
        return result

    def store_experience(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < Config.BATCH_SIZE:
            return None
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(Config.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(Config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(Config.DEVICE)
        next_states = torch.cat(next_states).to(Config.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32).to(Config.DEVICE)

        current_q = self.policy_net(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + Config.GAMMA * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path="territorial_brain.pth"):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "episodes_done": self.episodes_done,
        }, path)
        print(f"   💾 Model saved to {path}")

    def load_model(self, path="territorial_brain.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=Config.DEVICE)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]
            self.steps_done = checkpoint["steps_done"]
            self.episodes_done = checkpoint["episodes_done"]
            print(f"   📂 Model loaded from {path}")


# =============================================
# STEP 8: REWARD CALCULATOR
# =============================================

class RewardCalculator:
    """Scores actions: gained territory = good, lost = bad."""

    def __init__(self):
        self.preprocessor = GamePreprocessor()

    def calculate_reward(self, prev_screenshot, curr_screenshot, game_over=False, won=False):
        if game_over:
            return 100.0 if won else -50.0

        prev_analysis = self.preprocessor.extract_territory_colors(prev_screenshot)
        curr_analysis = self.preprocessor.extract_territory_colors(curr_screenshot)
        reward = 0.0

        if len(prev_analysis["percentages"]) > 0 and len(curr_analysis["percentages"]) > 0:
            territory_change = curr_analysis["percentages"][0] - prev_analysis["percentages"][0]
            if territory_change > 0:
                reward += territory_change * 2.0
            else:
                reward += territory_change * 3.0

        if curr_analysis["num_players"] < prev_analysis["num_players"]:
            reward += 10.0

        reward -= 0.1
        return reward


# =============================================
# STEP 9: FAKE GAME ENVIRONMENT (for testing)
# =============================================

class FakeGameEnvironment:
    def __init__(self, width=400, height=400):
        self.width = width
        self.height = height
        self.step_count = 0
        self.max_steps = 100
        self.player_territory = 0.1
        self.reset()

    def reset(self):
        self.step_count = 0
        self.player_territory = 0.1
        return self._generate_screenshot()

    def _generate_screenshot(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        colors = [[65,105,225],[220,20,60],[50,205,50],[255,165,0],[128,128,128]]
        block_size = 20
        for y in range(0, self.height, block_size):
            for x in range(0, self.width, block_size):
                if random.random() < self.player_territory:
                    color = colors[0]
                else:
                    color = random.choice(colors[1:])
                img[y:y+block_size, x:x+block_size] = color
        return Image.fromarray(img)

    def step(self, action):
        self.step_count += 1
        if action["action_type"] == "click":
            if random.random() < 0.6:
                self.player_territory = min(1.0, self.player_territory + 0.02)
            else:
                self.player_territory = max(0.01, self.player_territory - 0.01)
        elif action["action_type"] == "expand":
            if random.random() < 0.4:
                self.player_territory = min(1.0, self.player_territory + 0.05)
            else:
                self.player_territory = max(0.01, self.player_territory - 0.03)
        elif action["action_type"] == "defend":
            self.player_territory = min(1.0, self.player_territory + 0.01)

        done = self.step_count >= self.max_steps or self.player_territory >= 0.8
        won = self.player_territory >= 0.8
        reward = 100.0 if (done and won) else (-20.0 if done else (self.player_territory - 0.1) * 10)
        return self._generate_screenshot(), reward, done, {"territory": self.player_territory, "step": self.step_count, "won": won}


# =============================================
# STEP 10: TRAINING LOOP
# =============================================

def train_agent(num_episodes=50, verbose=True):
    print("\n" + "=" * 60)
    print("🎮 STARTING TRAINING")
    print("=" * 60)

    agent = GameAgent()
    env = FakeGameEnvironment()
    preprocessor = GamePreprocessor()
    all_rewards = []

    for episode in range(1, num_episodes + 1):
        screenshot = env.reset()
        state = preprocessor.process_screenshot(screenshot)
        episode_reward = 0
        episode_losses = []

        while True:
            action = agent.select_action(screenshot)
            next_screenshot, reward, done, info = env.step(action)
            next_state = preprocessor.process_screenshot(next_screenshot)
            agent.store_experience(state, action["action_index"], reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            state = next_state
            screenshot = next_screenshot
            episode_reward += reward
            if done:
                break

        agent.episodes_done += 1
        all_rewards.append(episode_reward)

        if episode % Config.TARGET_UPDATE == 0:
            agent.update_target_network()

        if verbose and episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            print(f"   Episode {episode}/{num_episodes} | Reward: {episode_reward:.1f} | "
                  f"Avg: {avg_reward:.1f} | Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Territory: {info['territory']*100:.0f}%")

    print("\n✅ TRAINING COMPLETE!")
    agent.save_model("territorial_brain.pth")
    return agent


if __name__ == "__main__":
    trained_agent = train_agent(num_episodes=50, verbose=True)
    print("\n🎉 Done! Model saved as 'territorial_brain.pth'")
