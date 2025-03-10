import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from Box2D import b2World, b2PolygonShape
from casualty import Casualty
from rubble import Rubble
from entity_manager import EntityManager


class DisasterResponseEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode=None, world_size=(100, 100),
                 num_casualties=5, num_rubble=8):
        """
        Initialize the disaster response environment

        Args:
            render_mode: 'human' for interactive visualization, 'rgb_array' for array output
            world_size: Size of the world in pixels
            num_casualties: Number of casualties to place in the environment
            num_rubble: Number of rubble obstacles to place in the environment
        """
        self.world_size = world_size
        self.render_mode = render_mode

        # Physics parameters
        self.scale = 30.0  # pixels per meter
        # Create world with no gravity (top-down view)
        self.world = b2World(gravity=(0, 0), doSleep=True)

        # Initialize entity manager
        self.entity_manager = EntityManager()

        # Entity counts
        self.num_casualties = num_casualties
        self.num_rubble = num_rubble

        # Distance parameters
        self.rescue_proximity = 2.0  # How close agent needs to be to rescue
        self.clearing_proximity = 2.0  # How close agent needs to be to clear rubble

        # Action space: [steering, acceleration, rescue_attempt, clear_rubble_attempt]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Observation space will be set in reset() after we know the
        # number of casualties and rubble pieces

        # Initialize rendering
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Clear the world
        for body in self.world.bodies:
            self.world.DestroyBody(body)

        # Clear the entity manager
        self.entity_manager.clear()

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

        # Create rubble obstacles first (before casualties)
        for _ in range(self.num_rubble):
            # Random clearing time between 5-20 steps
            clearing_time = self.np_random.integers(5, 21)
            rubble = Rubble(clearing_time=clearing_time)
            rubble.create_in_world(self.world, self.scale, self.np_random, self.world_size)
            self.entity_manager.add_rubble(rubble)

        # Create casualties
        for _ in range(self.num_casualties):
            # Random severity between 1-3
            severity = self.np_random.integers(1, 4)
            casualty = Casualty(severity=severity)
            casualty.create_in_world(self.world, self.scale, self.np_random, self.world_size)
            self.entity_manager.add_casualty(casualty)

        # Update observation space to accommodate casualties and rubble
        casualty_features = 6  # Each casualty has 6 features
        rubble_features = 6  # Each rubble has 6 features
        agent_features = 5  # Agent has 5 features

        total_features = agent_features + \
                         (casualty_features * len(self.entity_manager.get_casualties())) + \
                         (rubble_features * len(self.entity_manager.get_rubble()))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )

        # Initialize step count and currently clearing rubble reference
        self.steps = 0
        self.currently_clearing_rubble = None

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Unpack action
        steering = float(action[0])  # Range: [-1, 1]
        acceleration = float(action[1])  # Range: [-1, 1]
        rescue_attempt = float(action[2]) > 0.5  # Boolean conversion
        clear_rubble_attempt = float(action[3]) > 0.5  # Boolean conversion

        # Apply steering and acceleration to agent
        self.agent.angle += steering * 0.1
        force = (np.cos(self.agent.angle) * acceleration * 10,
                 np.sin(self.agent.angle) * acceleration * 10)
        self.agent.ApplyForceToCenter(force, True)

        # Simulate physics
        self.world.Step(1.0 / 30.0, 10, 10)

        # Small negative reward per step to encourage efficiency
        step_reward = -0.1

        # Update casualty rewards/health based on time
        for casualty in self.entity_manager.get_casualties():
            casualty.update_reward()

        # Handle rubble clearing
        rubble_reward = 0.0
        agent_position = (self.agent.position[0], self.agent.position[1])

        # Stop clearing rubble if the agent moves away or tries to clear a different piece
        currently_clearing_rubble = self.entity_manager.get_currently_clearing_rubble()
        if currently_clearing_rubble:
            distance_to_current = np.sqrt(
                (agent_position[0] - currently_clearing_rubble.body.position[0]) ** 2 +
                (agent_position[1] - currently_clearing_rubble.body.position[1]) ** 2
            )
            if distance_to_current > self.clearing_proximity or not clear_rubble_attempt:
                currently_clearing_rubble.stop_clearing()
                self.currently_clearing_rubble = None

        # Process clearing attempt
        if clear_rubble_attempt:
            # Check if already clearing a rubble
            if not currently_clearing_rubble:
                # Find the closest uncleared rubble within range
                closest_rubble, closest_distance = self.entity_manager.find_nearest_rubble(
                    agent_position, only_uncleared=True
                )

                # Start clearing the closest rubble if found and within proximity
                if closest_rubble and closest_distance <= self.clearing_proximity:
                    closest_rubble.start_clearing()
                    self.currently_clearing_rubble = closest_rubble

        # Update all rubble that is being cleared
        for rubble in self.entity_manager.get_rubble():
            if rubble.being_cleared:
                was_cleared = rubble.update_clearing()
                if was_cleared:
                    # Award a small reward for clearing rubble
                    rubble_reward += 5.0
                    # If this was the rubble we were clearing, reset reference
                    if rubble == self.currently_clearing_rubble:
                        self.currently_clearing_rubble = None

        # Handle rescue attempt if action indicates
        rescue_reward = 0.0
        if rescue_attempt:
            # Get unblocked casualties near the agent
            nearby_casualties = self.entity_manager.get_casualties_in_radius(
                agent_position, self.rescue_proximity,
                only_alive=True, only_unrescued=True
            )

            for casualty in nearby_casualties:
                # Only attempt rescue if casualty isn't blocked by rubble
                if not self.entity_manager.is_casualty_blocked_by_rubble(casualty):
                    rescue_success = casualty.attempt_rescue(
                        agent_position, self.rescue_proximity
                    )
                    if rescue_success:
                        # Award the current reward value if rescue successful
                        rescue_reward += casualty.current_reward

        # Calculate total reward
        reward = step_reward + rescue_reward + rubble_reward

        # Check termination conditions
        self.steps += 1

        # Episode is terminated if all casualties are either rescued or dead
        all_handled = (self.entity_manager.count_alive_casualties() == 0)
        terminated = all_handled

        # Time limit
        truncated = self.steps >= 1000  # Arbitrary time limit

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

        # Get casualty and rubble observations from entity manager
        casualty_obs, rubble_obs = self.entity_manager.get_entity_observations()

        # Combine agent, casualty, and rubble observations
        return np.array(agent_state + casualty_obs + rubble_obs, dtype=np.float32)

    def _get_info(self):
        # Get entity statistics from the entity manager
        entity_stats = self.entity_manager.get_entity_stats()

        return {
            'steps': self.steps,
            'casualties_rescued': entity_stats['casualties_rescued'],
            'casualties_alive': entity_stats['casualties_alive'],
            'casualties_dead': entity_stats['casualties_dead'],
            'rubble_cleared': entity_stats['rubble_cleared'],
            'total_rubble': entity_stats['rubble_total'],
            'currently_clearing': entity_stats['currently_clearing']
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
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        # Fill background
        self.screen.fill((255, 255, 255))

        # Draw rubble first
        for rubble in self.entity_manager.get_rubble():
            rubble.render(self.screen, self.scale)

        # Draw casualties
        for casualty in self.entity_manager.get_casualties():
            casualty.render(self.screen, self.scale)

        # Draw agent
        agent_position = (int(self.agent.position[0] * self.scale), int(self.agent.position[1] * self.scale))
        agent_angle = self.agent.angle

        # Create a triangle to represent the agent's direction
        points = [
            (agent_position[0] + int(1.0 * self.scale * np.cos(agent_angle)),
             agent_position[1] + int(1.0 * self.scale * np.sin(agent_angle))),
            (agent_position[0] + int(0.5 * self.scale * np.cos(agent_angle + 1.5)),
             agent_position[1] + int(0.5 * self.scale * np.sin(agent_angle + 1.5))),
            (agent_position[0] + int(0.5 * self.scale * np.cos(agent_angle - 1.5)),
             agent_position[1] + int(0.5 * self.scale * np.sin(agent_angle - 1.5)))
        ]
        pygame.draw.polygon(self.screen, (0, 0, 255), points)  # Blue

        # Status text
        info = self._get_info()
        status_text = (f"Steps: {self.steps} | Rescued: {info['casualties_rescued']} | "
                       f"Alive: {info['casualties_alive']} | Dead: {info['casualties_dead']} | ")
        text_surface = self.font.render(status_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        # Draw instructions
        instructions = "Esc: Exit"
        instr_surface = self.font.render(instructions, True, (0, 0, 0))
        self.screen.blit(instr_surface, (10, self.world_size[1] - 30))

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