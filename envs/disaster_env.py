import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from Box2D import b2World, b2PolygonShape, b2CircleShape, b2FixtureDef


class DisasterResponseEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode=None, world_size=(100, 100), num_casualties=5, num_rubble=10):
        super(DisasterResponseEnv, self).__init__()

        self.world_size = world_size
        self.num_casualties = num_casualties
        self.num_rubble = num_rubble
        self.render_mode = render_mode

        # Physics parameters
        self.scale = 30.0  # pixels per meter
        # Create world with no gravity (top-down view)
        self.world = b2World(gravity=(0, 0), doSleep=True)

        # Action space: [steering, acceleration, request_clearance, rescue_attempt]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space: agent state (position, velocity, orientation)
        # plus information about casualties and rubble
        # [agent_x, agent_y, agent_vx, agent_vy, agent_theta,
        #  casualties_x1, casualties_y1, ... casualties_xN, casualties_yN,
        #  rubble_x1, rubble_y1, ... rubble_xM, rubble_yM]
        obs_dim = 5 + 2 * num_casualties + 2 * num_rubble
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize rendering
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Clear the world
        for body in self.world.bodies:
            self.world.DestroyBody(body)

        # Create agent (rescue vehicle) - Dynamic body
        self.agent = self.world.CreateDynamicBody(
            position=(self.np_random.uniform(10, self.world_size[0] - 10) / self.scale,
                      self.np_random.uniform(10, self.world_size[1] - 10) / self.scale),
            angle=self.np_random.uniform(0, 2 * np.pi),
            linearDamping=0.5,
            angularDamping=0.5,
            userData={"type": "agent"}
        )

        # Add fixture to agent
        self.agent.CreateFixture(
            shape=b2PolygonShape(vertices=[
                (-1, -0.5), (1, -0.5), (1, 0.5), (-0.5, 0.5)
            ]),
            density=1.0,
            friction=0.3
        )

        # Create boundaries
        self.create_boundaries()

        # Create casualties
        self.casualties = []
        for _ in range(self.num_casualties):
            position = self.find_free_position()
            # Static body for casualties
            casualty = self.world.CreateStaticBody(
                position=position,
                userData={"type": "casualty", "rescued": False, "health": 100.0}
            )
            casualty.CreateFixture(
                shape=b2CircleShape(radius=0.5),
                isSensor=True  # Makes it not collide physically
            )
            self.casualties.append(casualty)

        # Create rubble
        self.rubble = []
        for _ in range(self.num_rubble):
            position = self.find_free_position()
            size = self.np_random.uniform(1.0, 3.0)
            # Static body for rubble
            rubble = self.world.CreateStaticBody(
                position=position,
                userData={"type": "rubble", "cleared": False, "size": size}
            )
            rubble.CreateFixture(
                shape=b2PolygonShape(box=(size, size)),
                friction=0.5
            )
            self.rubble.append(rubble)

        # Initialize step count and rescue count
        self.steps = 0
        self.rescued_casualties = 0

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Unpack action
        steering = float(action[0])  # Range: [-1, 1]
        acceleration = float(action[1])  # Range: [-1, 1]
        request_clearance = float(action[2]) > 0.5  # Boolean
        rescue_attempt = float(action[3]) > 0.5  # Boolean

        # Apply steering and acceleration to agent
        self.agent.angle += steering * 0.1
        force = (np.cos(self.agent.angle) * acceleration * 10,
                 np.sin(self.agent.angle) * acceleration * 10)
        self.agent.ApplyForceToCenter(force, True)

        # Simulate physics
        self.world.Step(1.0 / 30.0, 10, 10)

        # Handle rubble clearance
        reward = -0.1  # Small negative reward per step
        if request_clearance:
            # Find closest rubble
            closest_rubble = None
            closest_dist = float('inf')
            for rubble in self.rubble:
                if not rubble.userData["cleared"]:
                    dist = np.sqrt(
                        (rubble.position[0] - self.agent.position[0]) ** 2 +
                        (rubble.position[1] - self.agent.position[1]) ** 2
                    )
                    if dist < 3.0 and dist < closest_dist:  # Within 3 meters
                        closest_rubble = rubble
                        closest_dist = dist

            if closest_rubble:
                closest_rubble.userData["cleared"] = True
                reward += 1.0  # Reward for clearing rubble
            else:
                reward -= 0.5  # Penalty for requesting clearance with no nearby rubble

        # Handle rescue attempts
        if rescue_attempt:
            # Find closest casualty
            closest_casualty = None
            closest_dist = float('inf')
            for casualty in self.casualties:
                if not casualty.userData["rescued"]:
                    dist = np.sqrt(
                        (casualty.position[0] - self.agent.position[0]) ** 2 +
                        (casualty.position[1] - self.agent.position[1]) ** 2
                    )
                    if dist < 1.0 and dist < closest_dist:  # Within 1 meter
                        closest_casualty = casualty
                        closest_dist = dist

            if closest_casualty:
                closest_casualty.userData["rescued"] = True
                self.rescued_casualties += 1
                reward += 10.0  # Big reward for rescuing casualty
            else:
                reward -= 0.5  # Penalty for attempting rescue with no nearby casualties

        # Update casualty health (optional: casualties' health deteriorates over time)
        for casualty in self.casualties:
            if not casualty.userData["rescued"]:
                casualty.userData["health"] -= 0.1  # Health decreases over time

        # Check termination conditions
        self.steps += 1
        terminated = self.rescued_casualties == self.num_casualties

        # Check if agent is stuck or out of bounds
        agent_pos = self.agent.position
        if (agent_pos[0] < 0 or agent_pos[0] > self.world_size[0] / self.scale or
                agent_pos[1] < 0 or agent_pos[1] > self.world_size[1] / self.scale):
            terminated = True
            reward -= 10.0  # Penalty for going out of bounds

        # Time limit
        truncated = self.steps >= 1000  # Arbitrary time limit

        # Add bonus for completion
        if terminated and self.rescued_casualties == self.num_casualties:
            reward += 50.0

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()

        # Render if needed
        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Agent state: [x, y, vx, vy, theta]
        agent_state = [
            self.agent.position[0],
            self.agent.position[1],
            self.agent.linearVelocity[0],
            self.agent.linearVelocity[1],
            self.agent.angle
        ]

        # Casualty positions
        casualty_positions = []
        for casualty in self.casualties:
            if not casualty.userData["rescued"]:
                casualty_positions.extend([
                    casualty.position[0],
                    casualty.position[1]
                ])
            else:
                casualty_positions.extend([0, 0])  # Zeroed if rescued

        # Rubble positions
        rubble_positions = []
        for rubble in self.rubble:
            if not rubble.userData["cleared"]:
                rubble_positions.extend([
                    rubble.position[0],
                    rubble.position[1]
                ])
            else:
                rubble_positions.extend([0, 0])  # Zeroed if cleared

        return np.array(agent_state + casualty_positions + rubble_positions, dtype=np.float32)

    def _get_info(self):
        return {
            'steps': self.steps,
            'casualties_remaining': self.num_casualties - self.rescued_casualties,
            'rescued_casualties': self.rescued_casualties
        }

    def render(self):
        if self.render_mode is None:
            return

        # Initialize pygame if needed
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Disaster Response Simulation")
            self.screen = pygame.display.set_mode((self.world_size[0], self.world_size[1]))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

            # Process events once to ensure window is responsive
            pygame.event.pump()

        # Process events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Fill background
        self.screen.fill((255, 255, 255))

        # Draw rubble
        for rubble in self.rubble:
            if not rubble.userData["cleared"]:
                position = (int(rubble.position[0] * self.scale), int(rubble.position[1] * self.scale))
                size = int(rubble.userData["size"] * self.scale)
                pygame.draw.rect(
                    self.screen,
                    (100, 100, 100),  # Dark grey
                    pygame.Rect(position[0] - size, position[1] - size, 2 * size, 2 * size)
                )
            else:
                position = (int(rubble.position[0] * self.scale), int(rubble.position[1] * self.scale))
                size = int(rubble.userData["size"] * self.scale)
                pygame.draw.rect(
                    self.screen,
                    (200, 200, 200),  # Light grey
                    pygame.Rect(position[0] - size, position[1] - size, 2 * size, 2 * size)
                )

        # Draw casualties
        for casualty in self.casualties:
            position = (int(casualty.position[0] * self.scale), int(casualty.position[1] * self.scale))
            if not casualty.userData["rescued"]:
                pygame.draw.circle(
                    self.screen,
                    (255, 0, 0),  # Red
                    position,
                    int(0.5 * self.scale)
                )
            else:
                pygame.draw.circle(
                    self.screen,
                    (0, 255, 0),  # Green
                    position,
                    int(0.5 * self.scale)
                )

        # Draw agent
        agent_position = (int(self.agent.position[0] * self.scale), int(self.agent.position[1] * self.scale))
        agent_angle = self.agent.angle

        # Create a triangle to represent the agent's direction
        points = [
            (agent_position[0] + int(1.0 * self.scale * np.cos(agent_angle)),
             agent_position[1] + int(1.0 * self.scale * np.sin(agent_angle))),
            (agent_position[0] + int(0.5 * self.scale * np.cos(agent_angle + 2.5)),
             agent_position[1] + int(0.5 * self.scale * np.sin(agent_angle + 2.5))),
            (agent_position[0] + int(0.5 * self.scale * np.cos(agent_angle - 2.5)),
             agent_position[1] + int(0.5 * self.scale * np.sin(agent_angle - 2.5)))
        ]
        pygame.draw.polygon(self.screen, (0, 0, 255), points)  # Blue

        # Status text
        status_text = f"Steps: {self.steps}, Casualties: {self.rescued_casualties}/{self.num_casualties}"
        text_surface = self.font.render(status_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        # Draw instructions
        instructions = "ESC: Exit | R: Reset"
        instr_surface = self.font.render(instructions, True, (0, 0, 0))
        self.screen.blit(instr_surface, (10, self.world_size[1] - 40))

        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        # Ensure all events are processed
        pygame.event.pump()

        if self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def create_boundaries(self):
        # Create four walls around the environment
        wall_thickness = 1.0

        # Bottom wall
        bottom_wall = self.world.CreateStaticBody(
            position=(self.world_size[0] / (2 * self.scale), 0),
            userData={"type": "wall"}
        )
        bottom_wall.CreateFixture(
            shape=b2PolygonShape(box=(self.world_size[0] / (2 * self.scale), wall_thickness))
        )

        # Top wall
        top_wall = self.world.CreateStaticBody(
            position=(self.world_size[0] / (2 * self.scale), self.world_size[1] / self.scale),
            userData={"type": "wall"}
        )
        top_wall.CreateFixture(
            shape=b2PolygonShape(box=(self.world_size[0] / (2 * self.scale), wall_thickness))
        )

        # Left wall
        left_wall = self.world.CreateStaticBody(
            position=(0, self.world_size[1] / (2 * self.scale)),
            userData={"type": "wall"}
        )
        left_wall.CreateFixture(
            shape=b2PolygonShape(box=(wall_thickness, self.world_size[1] / (2 * self.scale)))
        )

        # Right wall
        right_wall = self.world.CreateStaticBody(
            position=(self.world_size[0] / self.scale, self.world_size[1] / (2 * self.scale)),
            userData={"type": "wall"}
        )
        right_wall.CreateFixture(
            shape=b2PolygonShape(box=(wall_thickness, self.world_size[1] / (2 * self.scale)))
        )

    def find_free_position(self):
        """Find a position that doesn't overlap with existing objects"""
        while True:
            position = (
                self.np_random.uniform(5, self.world_size[0] - 5) / self.scale,
                self.np_random.uniform(5, self.world_size[1] - 5) / self.scale
            )

            # Check distance from agent (if agent exists)
            if hasattr(self, 'agent'):
                agent_dist = np.sqrt(
                    (position[0] - self.agent.position[0]) ** 2 +
                    (position[1] - self.agent.position[1]) ** 2
                )
                if agent_dist < 3.0:
                    continue

            # Check distance from casualties (if any exist)
            if hasattr(self, 'casualties') and self.casualties:
                too_close = False
                for casualty in self.casualties:
                    dist = np.sqrt(
                        (position[0] - casualty.position[0]) ** 2 +
                        (position[1] - casualty.position[1]) ** 2
                    )
                    if dist < 2.0:
                        too_close = True
                        break
                if too_close:
                    continue

            # Check distance from rubble (if any exist)
            if hasattr(self, 'rubble') and self.rubble:
                too_close = False
                for rubble in self.rubble:
                    dist = np.sqrt(
                        (position[0] - rubble.position[0]) ** 2 +
                        (position[1] - rubble.position[1]) ** 2
                    )
                    if dist < 3.0:
                        too_close = True
                        break
                if too_close:
                    continue

            return position