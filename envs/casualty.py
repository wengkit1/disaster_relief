import numpy as np
import pygame
from Box2D import b2CircleShape
from entity import Entity


class Casualty(Entity):
    """
    Represents a casualty that needs to be rescued in the disaster environment.
    """

    def __init__(self, severity: int = 1):
        """
        Initialize a casualty

        Args:
            severity (int): Severity level of the casualty (1-3)
                            Higher severity means higher rescue priority & reward
        """
        self.body = None
        self.position = (0, 0)  # Initial position, will be set in create_in_world
        self.rescued = False  # Rescue status
        self.severity = severity  # Severity level

        self.current_reward = 10.0
        self.is_alive = True

    def create_in_world(self, world, scale: float, random_generator, world_size=None):
        """
        Create the casualty in the Box2D world

        Args:
            world: The Box2D world object
            scale: The pixel-to-meter scale factor
            random_generator: Random number generator for positioning
            world_size: Optional tuple (width, height) of world dimensions
        """
        # Generate random position based on world size
        if world_size:
            # Convert from pixels to Box2D meters
            world_width_meters = world_size[0] / scale
            world_height_meters = world_size[1] / scale

            # Keep away from edges with a 10% margin
            margin_x = world_width_meters * 0.1
            margin_y = world_height_meters * 0.1

            self.position = (
                random_generator.uniform(margin_x, world_width_meters - margin_x),
                random_generator.uniform(margin_y, world_height_meters - margin_y)
            )
        else:
            # Fallback to original behavior with fixed values if no world_size provided
            self.position = (
                random_generator.uniform(10, 90) / scale,
                random_generator.uniform(10, 90) / scale
            )

        # Create a static body for the casualty
        self.body = world.CreateStaticBody(
            position=self.position,
            userData={"type": "casualty", "instance": self}
        )

        # Add circular fixture as a sensor (doesn't physically interact)
        self.body.CreateFixture(
            shape=b2CircleShape(radius=1.0),
            isSensor=True,  # Doesn't cause physical collisions
        )

    def render(self, screen, scale: float):
        """
        Render the casualty on the screen as a colored circle

        Args:
            screen: Pygame screen object
            scale: The pixel-to-meter scale factor
        """
        # Don't render if already rescued
        if self.rescued:
            return

        # Convert Box2D position to screen pixels
        position = (int(self.body.position[0] * scale),
                    int(self.body.position[1] * scale))
        radius = int(1.0 * scale)

        # Outer circle (red for casualty)
        pygame.draw.circle(screen, (255, 0, 0), position, radius)

        # Inner circle (color varies with severity)
        color = (255, 255 - self.severity * 50, 0)  # Higher severity is more orange/red
        pygame.draw.circle(screen, color, position, radius // 2)

    def get_observation(self) -> np.ndarray:
        """
        Return the casualty's state as a numpy array

        Returns:
            np.ndarray: [x, y, rescued_status, severity]
        """
        return np.array([
            self.body.position[0],  # x-position
            self.body.position[1],  # y-position
            float(self.rescued),  # Rescue status (0.0 or 1.0)
            float(self.severity),
            self.current_reward,
            float(self.is_alive)
        ], dtype=np.float32)

    # Add a new method after the get_observation method:

    def update_reward(self, decay_rate: float = 0.01):
        """
        Update casualty's reward/health based on severity and time

        Args:
            decay_rate: Base rate at which reward decays per step

        Returns:
            bool: True if casualty is still alive, False if now dead
        """
        if self.rescued or not self.is_alive:
            return self.is_alive

        actual_decay = decay_rate * self.severity
        self.current_reward = max(0.0, self.current_reward - actual_decay)

        # Mark as dead if reward reaches zero
        if self.current_reward <= 0.0:
            self.is_alive = False
            self.current_reward = 0.0

        return self.is_alive

    def attempt_rescue(self, agent_position, proximity_threshold: float = 2.0) -> bool:
        """
        Try to rescue this casualty from the given agent position

        Args:
            agent_position: The position of the agent (x, y)
            proximity_threshold: How close the agent needs to be to rescue

        Returns:
            bool: True if rescue was successful, False otherwise
        """
        # Already rescued - nothing to do
        if self.rescued:
            return False

        # Calculate distance between agent and casualty
        dx = agent_position[0] - self.body.position[0]
        dy = agent_position[1] - self.body.position[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # If agent is close enough, rescue succeeds
        if distance <= proximity_threshold:
            self.rescued = True
            return True

        return False