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
ASTEROID_COUNT = 8 # Increased for more challenge
ASTEROID_SPEED_MIN = 0.5
ASTEROID_SPEED_MAX = 2



# Object type IDs for raycasting
TYPE_EMPTY = 0
TYPE_ASTEROID = 1
TYPE_TARGET = 2


# --- Game Object Classes (Mostly unchanged) ---

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

    def perform_action(self, action: int):
        """Sets thruster states based on the agent's chosen action."""
        if self.is_exploding: return
        # Action space: 0: None, 1: Thrust, 2: Turn Left, 3: Turn Right
        main_thruster_on = (action == 1)
        left_thruster_on = (action == 3) # Firing left thruster turns ship right
        right_thruster_on = (action == 2) # Firing right thruster turns ship left

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
            rad_angle = math.radians(self.angle+90)
            acc = pygame.math.Vector2(math.cos(rad_angle), -math.sin(rad_angle)) * THRUST_POWER
            self.vel += acc
            self.spawn_particles(side_angle=90, strength=5)

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
        self.screen =pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)


        try:
            rocket_img_orig = pygame.image.load("image/ship.png").convert_alpha()
            asteroid_img_orig = pygame.image.load("image/asteroid.png").convert_alpha()
            self.rocket_img = pygame.transform.scale(rocket_img_orig, (40, 55))
            self.asteroid_img = pygame.transform.scale(asteroid_img_orig, (70, 70))
        except pygame.error as e:
            print(str(e))
            print("Error loading images! Make sure 'ship.png' and 'asteroid.png' are in an 'image' folder.")
            pygame.quit()
            exit()
        self.action_space_n = 4


        n_ship_state = 7
        n_target_state = 2
        n_asteroid_state = ASTEROID_COUNT * 4
        self.observation_space_shape = (n_ship_state + n_target_state + n_asteroid_state,)

    def reset(self):
        """Resets the environment for a new episode."""
        self.player = Rocket(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, self.rocket_img)
        self.asteroids = [Asteroid(self.asteroid_img) for _ in range(ASTEROID_COUNT)]
        self.target = Target()
        self.score = 0
        self.steps = 0
        self.max_steps = 2500 # End episode if it takes too long
        self.distance_to_target = self.player.pos.distance_to(self.target.pos)

        return self._get_observation()

    def _get_observation(self):
            """
            Constructs the observation vector for the agent with global coordinates.
            All values are normalized to be roughly between -1 and 1.
            """
            state = []

            # --- 1. Ship's own state (normalized) ---
            angle_rad = math.radians(self.player.angle)
            state.extend([
                self.player.pos.x / SCREEN_WIDTH,
                self.player.pos.y / SCREEN_HEIGHT,
                self.player.vel.x / 10.0,  # Normalize velocity by a reasonable max
                self.player.vel.y / 10.0,
                math.cos(angle_rad),
                -math.sin(angle_rad),
                self.player.angle_vel/100 #normalize
            ])

            # --- 2. Target's state (normalized) ---
            state.extend([
                self.target.pos.x / SCREEN_WIDTH,
                self.target.pos.y / SCREEN_HEIGHT,
            ])

            # --- 3. Asteroids' state (normalized) ---
            for asteroid in self.asteroids:
                state.extend([
                    asteroid.pos.x / SCREEN_WIDTH,
                    asteroid.pos.y / SCREEN_HEIGHT,
                    asteroid.vel.x / 4.0, # Asteroids are slower, normalize by smaller max
                    asteroid.vel.y / 4.0,
                ])
            
            
            return np.array(state, dtype=np.float32)
    def step(self, action):
        self.steps += 1
        reward = 0
        done = False
        info = {}
        self.player.perform_action(action)
        self.player.update()
        for asteroid in self.asteroids: asteroid.update()

        # 1. Time penalty (encourages speed)
        reward -= 0.01 # Small penalty for every step taken.

        # 2. Distance-based reward (Simplified and more effective)
        new_dist = self.player.pos.distance_to(self.target.pos)
        # Reward is the amount of progress made toward the target.
        reward += (self.distance_to_target - new_dist) * 0.1 # Increased incentive
        distance_bonus = 5.0 / (new_dist + 1.0)
        reward += distance_bonus 
        self.distance_to_target = new_dist

        speed = self.player.vel.length()
        if speed < 0.1 and action == 0: # Si l'agent choisit de ne rien faire et qu'il est lent
            reward -= 0.1 # Grosse pénalité pour l'inactivité

        # --- 3. RE-ENABLE ALIGNMENT REWARD ---
        #vec_to_target = (self.target.pos - self.player.pos).normalize()
        # Use the same physics as the thruster for the forward vector
        #rocket_forward_vec = pygame.math.Vector2(math.sin(math.radians(self.player.angle)), -math.cos(math.radians(self.player.angle)))
        #alignment_reward = rocket_forward_vec.dot(vec_to_target)
        #reward += alignment_reward * 0.02 # Give a small, continuous reward for pointing correctly

        # --- Event-based rewards (large one-time rewards/penalties) ---
        if self.distance_to_target < self.player.radius + self.target.radius:
            reward += 100
            self.score += 1
            self.target.spawn()
            self.distance_to_target = self.player.pos.distance_to(self.target.pos)
            info['target_collected'] = True

        for asteroid in self.asteroids:
            if self.player.pos.distance_to(asteroid.pos) < self.player.radius + asteroid.radius:
                reward -= 100
                self.player.explode()
                done = True
                info['crashed'] = True
                break
                
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

# Use a GPU if available, otherwise use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# namedtuple for storing experiences in the Replay Buffer
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
    """The neural network that estimates Q-values."""
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128,128 )
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, env):
        # Hyperparameters
        self.BATCH_SIZE = 256
        self.GAMMA = 0.95
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.01 # Target network update rate
        self.LR = 3e-4 # Learning rate
        self.REPLAY_BUFFER_SIZE = 20000 # Store more diverse experiences.
        self.LEARNING_UPDATES_PER_STEP = 2 # Perform multiple learning steps for each action taken.

        self.replay_buffer = ReplayBuffer(self.REPLAY_BUFFER_SIZE)

        self.env = env
        self.n_actions = env.action_space_n
        n_observations = env.observation_space_shape[0]

        self.policy_net = QNetwork(n_observations, self.n_actions).to(device)
        self.target_net = QNetwork(n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.replay_buffer = ReplayBuffer(10000)
        self.steps_done = 0

    def choose_action(self, state):
        """Choose action using an epsilon-greedy policy."""
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
     
        if random.random() > eps_threshold:
            # Exploit: choose the best action from the policy network
            with torch.no_grad():
                # state is already a tensor with the correct shape [1, n_observations]
                action_values = self.policy_net(state)
                
                # --- FIX 1: Use argmax which is simpler and more robust ---
                # It directly returns the index of the max value.
                # We then .view(1, 1) to give it the batch and action dimensions
                # for the replay buffer.
                return torch.argmax(action_values).view(1, 1)
        else:
            # Explore: choose a random action
            # --- FIX 2: Select a random action from the valid range ---
            # self.env.action_space_n is 4, but valid actions are 0, 1, 2, 3.
            # random.randrange(n) gives an int from 0 to n-1.
            action = random.randrange(self.env.action_space_n)
            return torch.tensor([[action]], device=device, dtype=torch.long) 


    def learn(self):
        """Perform one step of the optimization."""
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return # Don't learn until we have enough experience

        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions)) # Converts batch-array of Transitions to Transition of batch-arrays

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
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
    EVALUATION_MODE = False # <-- SET TO True TO WATCH, False TO TRAIN
    env = RocketEnv()
    agent = DQNAgent(env)

        # --- AJOUT : Suivi des performances pour l'exploration intelligente ---
    scores_deque = deque(maxlen=100) # Stocke les 100 derniers scores
    best_avg_score = -np.inf # Garde en mémoire le meilleur score moyen
    episodes_since_improvement = 0

   # --- Load the model ---
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        # Note: We load the model onto the CPU first to avoid GPU memory issues
        # if the saving and loading environments are different.
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        
        if not EVALUATION_MODE:
            # If we are continuing to train, load everything
            agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.steps_done = checkpoint['steps_done']
    else:
        print("No pre-trained model found.")
        if EVALUATION_MODE:
            print("Cannot run in evaluation mode without a model. Exiting.")
            exit()

    if EVALUATION_MODE:
        # In evaluation, we don't need random actions or gradients
        agent.policy_net.eval()
        num_episodes = 20 # Run for 20 episodes to watch
        render_every = 1  # Render every episode
        save_every = 1000
    else:
        # In training mode
        agent.policy_net.train()
        num_episodes = 10000
        render_every = 10
        save_every = 25

    for i_episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        should_render = (i_episode + 1) % render_every == 0
        should_save = (i_episode + 1) % save_every == 0

        while True:
            # In eval mode, disable exploration by setting a high threshold
            if EVALUATION_MODE:
                with torch.no_grad():
                     action_values = agent.policy_net(state)
                     action = torch.argmax(action_values).view(1, 1)
            else:
                action = agent.choose_action(state)
               

            observation, reward, done, info = env.step(action.item())
            total_reward += reward
            
            # Don't do any learning if in evaluation mode
            if not EVALUATION_MODE:
                reward_tensor = torch.tensor([reward], device=device)
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
                agent.replay_buffer.push(state, action, next_state, reward_tensor)
                if len(agent.replay_buffer) > agent.BATCH_SIZE:
                    for _ in range(agent.LEARNING_UPDATES_PER_STEP):
                        agent.learn()
                
                # Soft update of the target network's weights
                agent.update_target_net()
            else: # In eval mode, just update the state
                 if not done:
                    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


            if should_render:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        exit()

            
            if done:
                print(f"Episode {i_episode+1} | Score: {env.score} | Total Reward: {total_reward:.2f}")
                if should_render:
                    pygame.time.wait(1000) 



                 # --- AJOUT : Logique de l'exploration intelligente ---
                scores_deque.append(env.score)
                current_avg_score = np.mean(scores_deque)

                if len(scores_deque) == 100: # Attendre d'avoir assez de données
                    if current_avg_score > best_avg_score:
                        best_avg_score = current_avg_score
                        episodes_since_improvement = 0
                        print(f"!!! Nouveau meilleur score moyen : {best_avg_score:.2f} !!!")
                    else:
                        episodes_since_improvement += 1

                # Si l'agent ne s'améliore pas depuis 200 épisodes, il est bloqué.
                if episodes_since_improvement > 200:
                    print("--- L'AGENT EST BLOQUÉ ! RÉINITIALISATION DE L'EXPLORATION. ---")
                    # On réinitialise sa curiosité pour le forcer à essayer de nouvelles choses
                    agent.steps_done = 0 
                    episodes_since_improvement = 0
                    best_avg_score = -np.inf # On réinitialise aussi le score de référence

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
        print(f"Saving checkpoint to {MODEL_PATH}...")
        torch.save({
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'target_net_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'steps_done': agent.steps_done,
        }, MODEL_PATH)

    env.close()