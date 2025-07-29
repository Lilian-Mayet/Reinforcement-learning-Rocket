import pygame
import math
import random

# --- Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
WHITE = (255,255,255)

# Physics Constants
THRUST_POWER = 0.2
ROTATION_SPEED = 0.2
FRICTION = 0.995  # A tiny bit of drag

# Game Constants
ASTEROID_COUNT = 1
ASTEROID_SPEED_MIN = 0.5
ASTEROID_SPEED_MAX = 2.0

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
        else:
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
        self.radius = 25  # Diameter of 50
        self.color = YELLOW
        self.spawn()

    def spawn(self):
        """Spawns the target at a random location, not too close to the edge."""
        padding = 50
        self.pos = pygame.math.Vector2(
            random.randint(padding, SCREEN_WIDTH - padding),
            random.randint(padding, SCREEN_HEIGHT - padding)
        )

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos, self.radius)

class Rocket:
    def __init__(self, x, y, image):
        self.original_image = image
        self.reset(x, y)
        self.particles = []

    def reset(self, x, y):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        # Pointing "up" is -90 degrees in a system where 0 is to the right
        self.angle = 0
        self.angle_vel = 0
        self.radius = self.original_image.get_width() / 2.5
        self.image = self.original_image
        self.rect = self.image.get_rect(center=self.pos)
        self.main_thruster_on = False
        self.left_thruster_on = False
        self.right_thruster_on = False
        self.is_exploding = False
        self.raycast_results = [] 

    def handle_input(self, keys):
        if self.is_exploding: return
        self.main_thruster_on = keys[pygame.K_UP] or keys[pygame.K_w]
        # Firing the left thruster makes the ship turn right
        self.left_thruster_on = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        # Firing the right thruster makes the ship turn left
        self.right_thruster_on = keys[pygame.K_LEFT] or keys[pygame.K_a]
        
    def update(self,asteroids, target):
        if not self.is_exploding:
            self.update_movement()
            self.cast_rays(asteroids, target) 
        self.update_particles()

    def update_movement(self):
        # --- Handle Rotation ---
        if self.left_thruster_on: # Turn right
            self.angle_vel -= ROTATION_SPEED
            self.spawn_particles(side_angle=0, strength=2, offset_mult=0.3) # Left thruster
        if self.right_thruster_on: # Turn left
            self.angle_vel += ROTATION_SPEED
            self.spawn_particles(side_angle=180, strength=2, offset_mult=0.3) # Right thruster

        # --- Handle Main Thrust ---
        if self.main_thruster_on:
            rad_angle = math.radians(self.angle-90)
            acc = pygame.math.Vector2(math.cos(rad_angle), math.sin(rad_angle)) * THRUST_POWER
            self.vel += acc
            self.spawn_particles(side_angle=90, strength=5) # Main thruster

        # --- Update Physics ---
        self.vel *= FRICTION
        self.angle_vel *= FRICTION
        self.pos += self.vel
        
        # Keep angle between 0 and 360. More robust than clipping.
        self.angle = (self.angle + self.angle_vel) % 360

        self.wrap_around_screen()
        
        # The angle for pygame's rotate is opposite to math angle
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.pos)

    def wrap_around_screen(self):
        if self.pos.x > SCREEN_WIDTH: self.pos.x = 0
        if self.pos.x < 0: self.pos.x = SCREEN_WIDTH
        if self.pos.y > SCREEN_HEIGHT: self.pos.y = 0
        if self.pos.y < 0: self.pos.y = SCREEN_HEIGHT

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

    def update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
    

    def cast_rays(self, asteroids, target):
        """Calculates ray intersections and stores them for drawing."""
        self.raycast_results.clear()
        
        num_rays = 25
        ray_length = 600
        fov = 60  # Field of view in degrees

        start_angle = self.angle-90 - fov / 2
        
        for i in range(num_rays):
            if num_rays > 1:
                ray_angle_deg = start_angle + i * fov / (num_rays - 1)
            else:
                ray_angle_deg = self.angle
                
            ray_angle_rad = math.radians(ray_angle_deg)
            end_pos = self.pos + pygame.math.Vector2(math.cos(ray_angle_rad), math.sin(ray_angle_rad)) * ray_length
            
            hit_color = WHITE
            closest_hit_dist = float('inf')

            # Check all potential collidable objects
            collidables = [(a.pos, a.radius, RED) for a in asteroids]
            collidables.append((target.pos, target.radius, YELLOW))

            for c_pos, c_radius, c_color in collidables:
                # Simple and fast line-circle intersection check by stepping along the ray
                for step in range(0, int(ray_length), 5):
                    point_on_ray = self.pos + pygame.math.Vector2(math.cos(ray_angle_rad), math.sin(ray_angle_rad)) * step
                    dist_to_obj = point_on_ray.distance_to(c_pos)
                    
                    if dist_to_obj < c_radius:
                        if step < closest_hit_dist:
                            closest_hit_dist = step
                            hit_color = c_color
                        break # Found the closest hit for this object, no need to check further along this ray

            self.raycast_results.append((self.pos, end_pos, hit_color))
            
    def draw(self, screen):
        for p in self.particles:
            p.draw(screen)
        
        #Draw the rays underneath the rocket
        for start, end, color in self.raycast_results:
            pygame.draw.line(screen, color, start, end, 1)
        
        if not self.is_exploding:
            screen.blit(self.image, self.rect)

    def explode(self):
        self.is_exploding = True
        for _ in range(50):
            vel = pygame.math.Vector2(random.uniform(-5, 5), random.uniform(-5, 5))
            lifespan = random.randint(30, 60)
            radius = random.uniform(2, 6)
            color = random.choice([GRAY, RED, ORANGE])
            self.particles.append(Particle(self.pos, vel, radius, color, lifespan))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Rocket & Asteroids")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    try:
        rocket_img = pygame.image.load("image/ship.png").convert_alpha()
        asteroid_img = pygame.image.load("image/asteroid.png").convert_alpha()
        rocket_img = pygame.transform.scale(rocket_img, (40, 55))
        asteroid_img = pygame.transform.scale(asteroid_img, (70, 70))
    except pygame.error as e:
        print("Error loading images! Make sure 'ship.png' and 'asteroid.png' are in the 'image' folder.")
        return

    player = Rocket(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, rocket_img)
    asteroids = [Asteroid(asteroid_img) for _ in range(ASTEROID_COUNT)]
    target = Target()
    
    score = 0
    reset_timer = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        player.handle_input(keys)
        
        player.update(asteroids,target)
        for asteroid in asteroids:
            asteroid.update()
        
        # --- Collision Detection ---
        if not player.is_exploding:
            # Check for asteroid collision
            for asteroid in asteroids:
                distance = player.pos.distance_to(asteroid.pos)
                if distance < player.radius + asteroid.radius:
                    player.explode()
                    reset_timer = pygame.time.get_ticks() + 2000
                    break
            
            # Check for target collection
            distance_to_target = player.pos.distance_to(target.pos)
            if distance_to_target < player.radius + target.radius:
                target.spawn()
                score += 1
        
        if player.is_exploding and pygame.time.get_ticks() > reset_timer:
            player.reset(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

        # --- Drawing ---
        screen.fill(BLACK)
        
        target.draw(screen)
        player.draw(screen)
        for asteroid in asteroids:
            asteroid.draw(screen)

        # Display info text
        score_text = font.render(f"Score: {score}", True, YELLOW)
        vel_text = font.render(f"Velocity: {player.vel.length():.2f}", True, YELLOW)
        screen.blit(score_text, (10, 10))
        screen.blit(vel_text, (10, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()