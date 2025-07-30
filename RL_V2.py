import pygame
import math
import random
import numpy as np

from collections import namedtuple, deque

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

MODEL_PATH = "rocket_dqn_policy.pth"
# --- Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0) # For target ray hits

# Physics Constants
THRUST_POWER = 0.2
ROTATION_SPEED = 0.2
FRICTION = 0.995  # A tiny bit of drag

# Game Constants
ASTEROID_COUNT = 3 # Reduced for easier learning
ASTEROID_SPEED_MIN = 0.3
ASTEROID_SPEED_MAX = 1.5

# Object type IDs for raycasting
TYPE_EMPTY = 0
TYPE_ASTEROID = 1
TYPE_TARGET = 2

# --- Game Object Classes ---

class Particle:
    """A single particle for thruster or explosion effects."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius = max(0, self.radius - 0.1)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos, self.radius)

class Asteroid:
    """An asteroid that drifts and rotates."""
    def __init__(self, image):
        self.original_image = image
        self.spawn()

    def spawn(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            self.pos = pygame.math.Vector2(random.randint(0, SCREEN_WIDTH), -self.original_image.get_height())
        elif edge == 'bottom':
            self.pos = pygame.math.Vector2(random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT + self.original_image.get_height())
        elif edge == 'left':
            self.pos = pygame.math.Vector2(-self.original_image.get_width(), random.randint(0, SCREEN_HEIGHT))
        else: # right
            self.pos = pygame.math.Vector2(SCREEN_WIDTH + self.original_image.get_width(), random.randint(0, SCREEN_HEIGHT))

        target_point = pygame.math.Vector2(random.randint(SCREEN_WIDTH * 0.25, SCREEN_WIDTH * 0.75),
                                           random.randint(SCREEN_HEIGHT * 0.25, SCREEN_HEIGHT * 0.75))
        speed = random.uniform(ASTEROID_SPEED_MIN, ASTEROID_SPEED_MAX)
        self.vel = (target_point - self.pos).normalize() * speed

        self.angle = 0
        self.rotation_speed = random.uniform(-1.5, 1.5)
        self.radius = self.original_image.get_width() / 2

    def update(self):
        self.pos += self.vel
        self.angle += self.rotation_speed

        # Despawn and respawn if too far off-screen
        if self.pos.x < -100 or self.pos.x > SCREEN_WIDTH + 100 or \
           self.pos.y < -100 or self.pos.y > SCREEN_HEIGHT + 100:
            self.spawn()

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.pos)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

class Target:
    """A target for the player to collect."""
    def __init__(self):
        self.radius = 25
        self.color = YELLOW
        self.spawn()

    def spawn(self):
        padding = 50
        self.pos = pygame.math.Vector2(
            random.randint(padding, SCREEN_WIDTH - padding),
            random.randint(padding, SCREEN_HEIGHT - padding)
        )

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos, self.radius)

class Rocket:
    """The agent's controllable rocket. Logic is simplified for RL."""
    def __init__(self, x, y, image):
        self.original_image = image
        self.reset(x, y)
        self.particles = []

    def reset(self, x, y):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = 0 # Pointing right
        self.angle_vel = 0
        self.radius = self.original_image.get_width() / 2.5
        self.image = self.original_image
        self.rect = self.image.get_rect(center=self.pos)
        self.is_exploding = False
        self.raycast_results_for_drawing = [] # For visualization only

    def perform_action(self, action_tuple):
        """Accept a tuple of actions (thrust, rotation) and apply physics."""
        thrust_action, rotation_action = action_tuple
        
        # Action space: 
        # thrust_action: 0=nothing, 1=thrust
        # rotation_action: 0=nothing, 1=left, 2=right
        
        main_thruster_on = (thrust_action == 1)
        left_thruster_on = (rotation_action == 1)  # Turn left
        right_thruster_on = (rotation_action == 2) # Turn right

        # --- Handle Rotation ---
        if left_thruster_on:
            self.angle_vel -= ROTATION_SPEED
            self.spawn_particles(side_angle=0, strength=2, offset_mult=0.3)
        if right_thruster_on:
            self.angle_vel += ROTATION_SPEED
            self.spawn_particles(side_angle=180, strength=2, offset_mult=0.3)

        # --- Handle Main Thrust ---
        if main_thruster_on:
            # Pygame angle is counter-clockwise, 0 is right. Math angle is standard.
            rad_angle = math.radians(self.angle)
            acc = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * THRUST_POWER
            self.vel += acc
            self.spawn_particles(side_angle=180, strength=5)

    def update(self):
        """Updates physics and particles."""
        if not self.is_exploding:
            self.vel *= FRICTION
            self.angle_vel *= FRICTION
            self.pos += self.vel
            self.angle = (self.angle + self.angle_vel) % 360
            self.wrap_around_screen()
            self.image = pygame.transform.rotate(self.original_image, self.angle)
            self.rect = self.image.get_rect(center=self.pos)

        self.update_particles()

    def wrap_around_screen(self):
        if self.pos.x > SCREEN_WIDTH: self.pos.x = 0
        if self.pos.x < 0: self.pos.x = SCREEN_WIDTH
        if self.pos.y > SCREEN_HEIGHT: self.pos.y = 0
        if self.pos.y < 0: self.pos.y = SCREEN_HEIGHT
    
    def explode(self):
        if self.is_exploding: return
        self.is_exploding = True
        for _ in range(50):
            vel = pygame.math.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
            lifespan = random.randint(30, 60)
            radius = random.uniform(2, 6)
            color = random.choice([GRAY, RED, ORANGE])
            self.particles.append(Particle(self.pos, vel, radius, color, lifespan))

    def update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles: p.update()

    def draw(self, screen):
        # Draw rays first, so they are underneath the rocket
        for start, end, color in self.raycast_results_for_drawing:
            pygame.draw.line(screen, color, start, end, 1)

        # Draw particles
        for p in self.particles: p.draw(screen)

        if not self.is_exploding:
            screen.blit(self.image, self.rect)

    def spawn_particles(self, side_angle, strength, offset_mult=1.0):
        # Calculate angle relative to the rocket's current direction
        angle_rad = math.radians(self.angle + side_angle)
        direction = pygame.math.Vector2(math.cos(angle_rad), math.sin(angle_rad))
        
        # Position particles at the edge of the rocket
        spawn_pos = self.pos + direction * (self.original_image.get_height() / 2 * offset_mult)

        for _ in range(strength):
            # Particle velocity is in the thruster direction, with randomness
            particle_vel = direction * random.uniform(2, 4) + pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
            lifespan = random.randint(15, 30)
            radius = random.uniform(2, 5)
            self.particles.append(Particle(spawn_pos, particle_vel, radius, ORANGE, lifespan))

# --- RL Environment ---

class RocketEnv:
    """The Reinforcement Learning Environment."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Create simple colored rectangles if images don't exist
        try:
            rocket_img_orig = pygame.image.load("image/ship.png").convert_alpha()
            asteroid_img_orig = pygame.image.load("image/asteroid.png").convert_alpha()
            self.rocket_img = pygame.transform.scale(rocket_img_orig, (40, 55))
            self.asteroid_img = pygame.transform.scale(asteroid_img_orig, (70, 70))
        except pygame.error:
            print("Images not found, creating simple colored shapes")
            self.rocket_img = pygame.Surface((40, 55))
            self.rocket_img.fill(WHITE)
            pygame.draw.polygon(self.rocket_img, RED, [(20, 0), (0, 55), (40, 55)])
            
            self.asteroid_img = pygame.Surface((70, 70))
            self.asteroid_img.fill(GRAY)
            pygame.draw.circle(self.asteroid_img, (100, 100, 100), (35, 35), 35)

        self.action_space_n = 4

        n_ship_state = 7
        n_target_state = 3  # Added distance to target
        n_asteroid_state = ASTEROID_COUNT * 5  # Added distance to each asteroid
        self.observation_space_shape = (n_ship_state + n_target_state + n_asteroid_state,)

    def reset(self):
        """Resets the environment for a new episode."""
        self.player = Rocket(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, self.rocket_img)
        self.asteroids = [Asteroid(self.asteroid_img) for _ in range(ASTEROID_COUNT)]
        self.target = Target()
        self.score = 0
        self.steps = 0
        self.max_steps = 1500  # Reduced for faster training episodes
        self.distance_to_target = self.player.pos.distance_to(self.target.pos)
        self.previous_distance_to_target = self.distance_to_target

        return self._get_observation()

    def _get_observation(self):
        """
        Constructs the observation vector for the agent with normalized values.
        """
        state = []

        # --- 1. Ship's own state (normalized) ---
        angle_rad = math.radians(self.player.angle)
        state.extend([
            (self.player.pos.x - SCREEN_WIDTH/2) / (SCREEN_WIDTH/2),  # Centered normalization
            (self.player.pos.y - SCREEN_HEIGHT/2) / (SCREEN_HEIGHT/2),
            self.player.vel.x / 5.0,  # Normalize velocity
            self.player.vel.y / 5.0,
            math.cos(angle_rad),  # Direction as unit vector
            math.sin(angle_rad),
            self.player.angle_vel / 10.0  # Normalize angular velocity
        ])

        # --- 2. Target's state (normalized + distance) ---
        target_dx = (self.target.pos.x - self.player.pos.x) / SCREEN_WIDTH
        target_dy = (self.target.pos.y - self.player.pos.y) / SCREEN_HEIGHT
        target_distance = self.distance_to_target / (SCREEN_WIDTH + SCREEN_HEIGHT)  # Normalize distance
        
        state.extend([target_dx, target_dy, target_distance])

        # --- 3. Asteroids' state (normalized + distances) ---
        for asteroid in self.asteroids:
            ast_dx = (asteroid.pos.x - self.player.pos.x) / SCREEN_WIDTH
            ast_dy = (asteroid.pos.y - self.player.pos.y) / SCREEN_HEIGHT
            ast_vel_x = asteroid.vel.x / 2.0
            ast_vel_y = asteroid.vel.y / 2.0
            ast_distance = self.player.pos.distance_to(asteroid.pos) / (SCREEN_WIDTH + SCREEN_HEIGHT)
            
            state.extend([ast_dx, ast_dy, ast_vel_x, ast_vel_y, ast_distance])
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0
        done = False
        info = {}
        
        self.player.perform_action(action)
        self.player.update()
        for asteroid in self.asteroids: 
            asteroid.update()

        # Update distances
        self.previous_distance_to_target = self.distance_to_target
        self.distance_to_target = self.player.pos.distance_to(self.target.pos)

        # --- REWARD STRUCTURE ---
        
        # 1. Small time penalty to encourage efficiency
        reward -= 0.001
        
        # 2. Progress reward - reward getting closer to target
        progress = self.previous_distance_to_target - self.distance_to_target
        reward += progress * 0.01  # Scale factor for progress reward
        
        # 3. Proximity bonus - closer to target = higher reward
        max_distance = math.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
        proximity_bonus = (max_distance - self.distance_to_target) / max_distance * 0.1
        reward += proximity_bonus
        
        # 4. Velocity alignment reward
        if self.distance_to_target > 0:
            target_direction = (self.target.pos - self.player.pos).normalize()
            velocity_alignment = 0
            if self.player.vel.length() > 0.1:  # Only if moving
                velocity_direction = self.player.vel.normalize()
                velocity_alignment = target_direction.dot(velocity_direction)
            reward += velocity_alignment * 0.05

        # --- Event-based rewards ---
        
        # Target collection
        if self.distance_to_target < self.player.radius + self.target.radius:
            reward += 100  # Large positive reward
            self.score += 1
            self.target.spawn()
            self.distance_to_target = self.player.pos.distance_to(self.target.pos)
            self.previous_distance_to_target = self.distance_to_target
            info['target_collected'] = True

        # Asteroid collision
        for asteroid in self.asteroids:
            if self.player.pos.distance_to(asteroid.pos) < self.player.radius + asteroid.radius:
                reward -= 50  # Large negative reward
                self.player.explode()
                done = True
                info['crashed'] = True
                break

        # Asteroid proximity penalty (to encourage avoidance)
        min_asteroid_distance = float('inf')
        for asteroid in self.asteroids:
            dist = self.player.pos.distance_to(asteroid.pos)
            min_asteroid_distance = min(min_asteroid_distance, dist)
        
        if min_asteroid_distance < 100:  # If too close to any asteroid
            proximity_penalty = (100 - min_asteroid_distance) / 100 * 0.5
            reward -= proximity_penalty
                
        if self.steps >= self.max_steps:
            done = True
            info['timeout'] = True

        observation = self._get_observation()
        return observation, reward, done, info

    def render(self):
        """Draws the environment to the screen."""
        if self.screen is None:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("RL Rocket")

        self.screen.fill(BLACK)

        self.target.draw(self.screen)
        for asteroid in self.asteroids:
            asteroid.draw(self.screen)
        self.player.draw(self.screen)

        # Display info text
        score_text = self.font.render(f"Score: {self.score}", True, YELLOW)
        steps_text = self.font.render(f"Steps: {self.steps}", True, YELLOW)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

# --- NEURAL NETWORK AND AGENT ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    """A cyclic buffer to store transitions."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Return a random sample of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, n_observations):
        super(QNetwork, self).__init__()
        
        # Shared body that processes the state
        self.body = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Head 1: Thrust (2 actions: 0=nothing, 1=thrust)
        self.thrust_head = nn.Linear(64, 2)
        
        # Head 2: Rotation (3 actions: 0=nothing, 1=left, 2=right)
        self.rotation_head = nn.Linear(64, 3)

    def forward(self, x):
        body_output = self.body(x)
        thrust_values = self.thrust_head(body_output)
        rotation_values = self.rotation_head(body_output)
        return thrust_values, rotation_values

class DQNAgent:
    def __init__(self, env):
        # Hyperparameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.995  # Slightly higher discount factor
        self.EPS_START = 1.0
        self.EPS_END = 0.02
        self.EPS_DECAY = 10000  # Slower decay for more exploration
        self.TAU = 0.005  # Slower target network update
        self.LR = 1e-4  # Lower learning rate
        self.REPLAY_BUFFER_SIZE = 50000
        self.LEARNING_UPDATES_PER_STEP = 1
        self.MIN_BUFFER_SIZE = 2000  # Wait for more experiences before learning

        self.replay_buffer = ReplayBuffer(self.REPLAY_BUFFER_SIZE)
        self.env = env
        n_observations = env.observation_space_shape[0]

        self.policy_net = QNetwork(n_observations).to(device)
        self.target_net = QNetwork(n_observations).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.steps_done = 0

    def choose_action(self, state):
        """Choose action for each branch using epsilon-greedy."""
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
     
        if random.random() > eps_threshold:
            # EXPLOITATION
            with torch.no_grad():
                thrust_q_values, rotation_q_values = self.policy_net(state)
                thrust_action = torch.argmax(thrust_q_values).view(1, 1)
                rotation_action = torch.argmax(rotation_q_values).view(1, 1)
        else:
            # EXPLORATION
            thrust_action = torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)
            rotation_action = torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)
            
        return thrust_action, rotation_action

    def learn(self):
        """Perform one optimization step with separate losses for each branch."""
        if len(self.replay_buffer) < self.MIN_BUFFER_SIZE:
            return

        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward)
        
        # Unpack actions
        thrust_action_batch = torch.cat([a[0] for a in batch.action])
        rotation_action_batch = torch.cat([a[1] for a in batch.action])

        # Compute Q(s_t, a) for each branch
        thrust_q_values, rotation_q_values = self.policy_net(state_batch)
        
        current_thrust_q = thrust_q_values.gather(1, thrust_action_batch)
        current_rotation_q = rotation_q_values.gather(1, rotation_action_batch)

        # Compute V(s_{t+1}) for each branch
        next_thrust_q_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_rotation_q_values = torch.zeros(self.BATCH_SIZE, device=device)
        
        with torch.no_grad():
            if len(non_final_next_states) > 0:  # Check if we have valid next states
                next_thrust_raw, next_rotation_raw = self.target_net(non_final_next_states)
                next_thrust_q_values[non_final_mask] = next_thrust_raw.max(1)[0]
                next_rotation_q_values[non_final_mask] = next_rotation_raw.max(1)[0]
        
        # Compute expected Q values
        expected_thrust_q = (next_thrust_q_values * self.GAMMA) + reward_batch
        expected_rotation_q = (next_rotation_q_values * self.GAMMA) + reward_batch

        # Compute losses
        criterion = nn.SmoothL1Loss()
        thrust_loss = criterion(current_thrust_q, expected_thrust_q.unsqueeze(1))
        rotation_loss = criterion(current_rotation_q, expected_rotation_q.unsqueeze(1))
        
        total_loss = thrust_loss + rotation_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """Soft update of the target network's weights."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

if __name__ == "__main__":
    EVALUATION_MODE = False  # Set to True to watch, False to train
    env = RocketEnv()
    agent = DQNAgent(env)

    # Load the model if it exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        
        if not EVALUATION_MODE:
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.steps_done = checkpoint['steps_done']
    else:
        print("No pre-trained model found.")
        if EVALUATION_MODE:
            print("Cannot run in evaluation mode without a model. Exiting.")
            exit()

    if EVALUATION_MODE:
        agent.policy_net.eval()
        num_episodes = 20
        render_every = 1
        save_every = 1000
    else:
        agent.policy_net.train()
        num_episodes = 5000
        render_every = 50  # Render less frequently during training
        save_every = 100

    # Training statistics
    episode_rewards = []
    episode_scores = []

    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        should_render = (i_episode + 1) % render_every == 0
        should_save = (i_episode + 1) % save_every == 0

        while True:
            if EVALUATION_MODE:
                with torch.no_grad():
                    thrust_q_values, rotation_q_values = agent.policy_net(state)
                    thrust_action = torch.argmax(thrust_q_values).item()
                    rotation_action = torch.argmax(rotation_q_values).item()
                    thrust_action_tensor = torch.tensor([[thrust_action]], device=device, dtype=torch.long)
                    rotation_action_tensor = torch.tensor([[rotation_action]], device=device, dtype=torch.long)
            else:
                thrust_action_tensor, rotation_action_tensor = agent.choose_action(state)
                thrust_action = thrust_action_tensor.item()
                rotation_action = rotation_action_tensor.item()
            
            action_tuple = (thrust_action, rotation_action)
            observation, reward, done, info = env.step(action_tuple)
            total_reward += reward

            if not EVALUATION_MODE:
                reward_tensor = torch.tensor([reward], device=device)
                action_for_buffer = (thrust_action_tensor, rotation_action_tensor)
                
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
                agent.replay_buffer.push(state, action_for_buffer, next_state, reward_tensor)
                state = next_state
                
                # Learn from experiences
                if len(agent.replay_buffer) > agent.MIN_BUFFER_SIZE:
                    for _ in range(agent.LEARNING_UPDATES_PER_STEP):
                        agent.learn()
                
                agent.update_target_net()
            else:
                if not done:
                    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            if should_render:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        exit()
            
            if done:
                episode_rewards.append(total_reward)
                episode_scores.append(env.score)
                
                # Print statistics
                if len(episode_rewards) >= 100:
                    avg_reward = sum(episode_rewards[-100:]) / 100
                    avg_score = sum(episode_scores[-100:]) / 100
                    eps_threshold = agent.EPS_END + (agent.EPS_START - agent.EPS_END) * \
                        math.exp(-1. * agent.steps_done / agent.EPS_DECAY)
                    print(f"Episode {i_episode+1} | Score: {env.score} | Reward: {total_reward:.2f} | "
                          f"Avg100 Reward: {avg_reward:.2f} | Avg100 Score: {avg_score:.2f} | "
                          f"Epsilon: {eps_threshold:.3f}")
                else:
                    print(f"Episode {i_episode+1} | Score: {env.score} | Total Reward: {total_reward:.2f}")
                
                if should_render:
                    pygame.time.wait(1000)

                if should_save and not EVALUATION_MODE:
                    print(f"--- Saving checkpoint at episode {i_episode+1} ---")
                    torch.save({
                        'policy_net_state_dict': agent.policy_net.state_dict(),
                        'target_net_state_dict': agent.target_net.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'steps_done': agent.steps_done,
                    }, MODEL_PATH)
                break

    # Only save if we were training
    if not EVALUATION_MODE:
        print('Training complete')
        print(f"Saving final checkpoint to {MODEL_PATH}...")
        torch.save({
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'target_net_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'steps_done': agent.steps_done,
        }, MODEL_PATH)

    env.close()