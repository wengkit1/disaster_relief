import numpy as np
import pygame
from Box2D import b2PolygonShape
from entity import Entity


class Rubble(Entity):
    """
    Represents rubble obstacles in the disaster environment that block paths and
    may trap casualties. Rubble can be cleared by emergency services, but this takes time.
    """

    def __init__(self, clearing_time: int = 10):
        """
        Initialize a rubble obstacle

        Args:
            clearing_time (int): Time steps required to clear this rubble
        """
        self.body = None
        self.position = (0, 0)  # Initial position, will be set in create_in_world
        self.size = (0, 0)  # Size in meters, will be set in create_in_world
        self.cleared = False  # Whether the rubble has been cleared
        self.clearing_time = clearing_time  # Time steps needed to clear
        self.clearing_progress = 0  # Current progress in clearing (0 to clearing_time)
        self.being_cleared = False  # Flag to indicate if currently being cleared

    def create_in_world(self, world, scale: float, random_generator, world_size=None):
        """
        Create the rubble in the Box2D world

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

        # Random size for the rubble - make it an obstacle that's sizeable but not too large
        # Width between 2-5 meters, height between 2-5 meters
        self.size = (
            random_generator.uniform(2.0, 5.0),
            random_generator.uniform(2.0, 5.0)
        )

        # Create a static body for the rubble
        self.body = world.CreateStaticBody(
            position=self.position,
            userData={"type": "rubble", "instance": self}
        )

        # Add a rectangular fixture that causes physical collisions when not cleared
        self.fixture = self.body.CreateFixture(
            shape=b2PolygonShape(box=(self.size[0] / 2, self.size[1] / 2)),
            density=1.0,
            friction=0.3
        )

    def render(self, screen, scale: float):
        """
        Render the rubble on the screen

        Args:
            screen: Pygame screen object
            scale: The pixel-to-meter scale factor
        """
        # Don't render if cleared
        if self.cleared:
            return

        # Convert Box2D position and size to screen pixels
        position = (int(self.body.position[0] * scale),
                    int(self.body.position[1] * scale))
        size = (int(self.size[0] * scale), int(self.size[1] * scale))

        # Rectangle for the rubble
        rect = pygame.Rect(
            position[0] - size[0] // 2,
            position[1] - size[1] // 2,
            size[0],
            size[1]
        )

        # Pick color based on clearing status
        if self.being_cleared:
            # Gradient from brown to green based on clearing progress
            progress_ratio = self.clearing_progress / self.clearing_time
            red = int(139 * (1 - progress_ratio))
            green = int(69 * (1 - progress_ratio) + 200 * progress_ratio)
            blue = int(19 * (1 - progress_ratio))
            color = (red, green, blue)
        else:
            # Brown for normal rubble
            color = (150, 150, 150)  # Brown

        # Draw the rubble
        pygame.draw.rect(screen, color, rect)

        # Draw clearing progress if being cleared
        if self.being_cleared:
            # Draw progress bar above the rubble
            progress_width = int(size[0] * (self.clearing_progress / self.clearing_time))
            progress_rect = pygame.Rect(
                position[0] - size[0] // 2,
                position[1] - size[1] // 2 - 10,
                progress_width,
                5
            )
            pygame.draw.rect(screen, (0, 255, 0), progress_rect)  # Green progress bar

            # Draw the full bar outline
            full_rect = pygame.Rect(
                position[0] - size[0] // 2,
                position[1] - size[1] // 2 - 10,
                size[0],
                5
            )
            pygame.draw.rect(screen, (255, 255, 255), full_rect, 1)  # White outline

    def get_observation(self) -> np.ndarray:
        """
        Return the rubble's state as a numpy array

        Returns:
            np.ndarray: [x, y, width, height, cleared_status, clearing_progress]
        """
        return np.array([
            self.body.position[0],  # x-position
            self.body.position[1],  # y-position
            self.size[0],  # width
            self.size[1],  # height
            float(self.cleared),  # Cleared status (0.0 or 1.0)
            self.clearing_progress / self.clearing_time  # Normalized clearing progress
        ], dtype=np.float32)

    def start_clearing(self):
        """Start the process of clearing this rubble"""
        if not self.cleared and not self.being_cleared:
            self.being_cleared = True
            self.clearing_progress = 0

    def stop_clearing(self):
        """Stop the process of clearing this rubble"""
        self.being_cleared = False

    def update_clearing(self) -> bool:
        """
        Update the clearing progress if this rubble is being cleared

        Returns:
            bool: True if rubble is now cleared, False otherwise
        """
        if self.cleared or not self.being_cleared:
            return self.cleared

        self.clearing_progress += 1

        # Check if clearing is complete
        if self.clearing_progress >= self.clearing_time:
            self.cleared = True
            self.being_cleared = False

            # Make the fixture a sensor (no physical collision) once cleared
            self.body.DestroyFixture(self.fixture)
            self.fixture = self.body.CreateFixture(
                shape=b2PolygonShape(box=(self.size[0] / 2, self.size[1] / 2)),
                isSensor=True,  # No physical collision when cleared
            )

        return self.cleared

    def is_position_blocked(self, position, proximity_threshold: float = 0.5) -> bool:
        """
        Check if a given position is blocked by this rubble

        Args:
            position: The position to check (x, y)
            proximity_threshold: Additional buffer distance to consider

        Returns:
            bool: True if position is blocked, False otherwise
        """
        if self.cleared:
            return False

        # Check if position is within the rubble's bounds (plus threshold)
        x_min = self.body.position[0] - self.size[0] / 2 - proximity_threshold
        x_max = self.body.position[0] + self.size[0] / 2 + proximity_threshold
        y_min = self.body.position[1] - self.size[1] / 2 - proximity_threshold
        y_max = self.body.position[1] + self.size[1] / 2 + proximity_threshold

        return (x_min <= position[0] <= x_max and
                y_min <= position[1] <= y_max)