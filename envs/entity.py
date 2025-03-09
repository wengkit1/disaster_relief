from abc import ABC, abstractmethod
import numpy as np


class Entity(ABC):
    """
    Abstract base class for all entities in the disaster environment.
    All entities must implement these methods.
    """

    @abstractmethod
    def create_in_world(self, world, scale: float, random_generator, world_size=None):
        """
        Create the entity in the Box2D world

        Args:
            world: The Box2D world object
            scale: The pixel-to-meter scale factor
            random_generator: Random number generator for positioning
            world_size: Optional tuple (width, height) of world dimensions
        """
        pass

    @abstractmethod
    def render(self, screen, scale: float):
        """
        Render the entity on the screen

        Args:
            screen: Pygame screen object
            scale: The pixel-to-meter scale factor
        """
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """
        Return the entity's state as a numpy array for the observation space

        Returns:
            np.ndarray: Entity state information
        """
        pass