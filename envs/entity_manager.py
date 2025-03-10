import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union


class EntityManager:
    """
    Manages rubble and casualties in the disaster environment.
    Provides efficient spatial queries and observation access.
    """

    def __init__(self):
        """Initialize the entity manager"""
        # Collections of entities by type
        self.casualties = []
        self.rubble = []

        # Spatial lookup cache
        self._spatial_cache_valid = False

    def clear(self):
        """Clear all entity references"""
        self.casualties = []
        self.rubble = []
        self._spatial_cache_valid = False

    def add_casualty(self, casualty):
        """
        Add a casualty to the manager

        Args:
            casualty: The casualty entity to add
        """
        self.casualties.append(casualty)
        self._spatial_cache_valid = False

    def add_rubble(self, rubble):
        """
        Add a rubble obstacle to the manager

        Args:
            rubble: The rubble entity to add
        """
        self.rubble.append(rubble)
        self._spatial_cache_valid = False

    def get_casualties(self):
        """Get all casualty entities"""
        return self.casualties

    def get_rubble(self):
        """Get all rubble entities"""
        return self.rubble

    def find_nearest_casualty(self, position: Tuple[float, float],
                              only_alive=True, only_unrescued=True) -> Tuple[Optional[Any], float]:
        """
        Find the nearest casualty to a position

        Args:
            position: Position (x, y) to search from
            only_alive: Only consider alive casualties
            only_unrescued: Only consider unrescued casualties

        Returns:
            Tuple of (nearest casualty, distance) or (None, float('inf')) if none found
        """
        candidates = [c for c in self.casualties
                      if (not only_alive or c.is_alive) and
                      (not only_unrescued or not c.rescued)]

        if not candidates:
            return None, float('inf')

        nearest = None
        min_dist = float('inf')

        for casualty in candidates:
            dx = casualty.body.position[0] - position[0]
            dy = casualty.body.position[1] - position[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < min_dist:
                min_dist = dist
                nearest = casualty

        return nearest, min_dist

    def find_nearest_rubble(self, position: Tuple[float, float],
                            only_uncleared=True) -> Tuple[Optional[Any], float]:
        """
        Find the nearest rubble to a position

        Args:
            position: Position (x, y) to search from
            only_uncleared: Only consider uncleared rubble

        Returns:
            Tuple of (nearest rubble, distance) or (None, float('inf')) if none found
        """
        candidates = [r for r in self.rubble if (not only_uncleared or not r.cleared)]

        if not candidates:
            return None, float('inf')

        nearest = None
        min_dist = float('inf')

        for rubble in candidates:
            dx = rubble.body.position[0] - position[0]
            dy = rubble.body.position[1] - position[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist < min_dist:
                min_dist = dist
                nearest = rubble

        return nearest, min_dist

    def get_casualties_in_radius(self, position: Tuple[float, float], radius: float,
                                 only_alive=True, only_unrescued=True) -> List:
        """
        Find all casualties within a radius of a position

        Args:
            position: Center position (x, y) for the search
            radius: Search radius
            only_alive: Only consider alive casualties
            only_unrescued: Only consider unrescued casualties

        Returns:
            List of casualties within the radius
        """
        results = []

        for casualty in self.casualties:
            if (only_alive and not casualty.is_alive) or (only_unrescued and casualty.rescued):
                continue

            dx = casualty.body.position[0] - position[0]
            dy = casualty.body.position[1] - position[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist <= radius:
                results.append((casualty, dist))

        # Sort by distance
        results.sort(key=lambda x: x[1])
        return [c for c, _ in results]

    def get_rubble_in_radius(self, position: Tuple[float, float], radius: float,
                             only_uncleared=True) -> List:
        """
        Find all rubble within a radius of a position

        Args:
            position: Center position (x, y) for the search
            radius: Search radius
            only_uncleared: Only consider uncleared rubble

        Returns:
            List of rubble within the radius
        """
        results = []

        for rubble in self.rubble:
            if only_uncleared and rubble.cleared:
                continue

            dx = rubble.body.position[0] - position[0]
            dy = rubble.body.position[1] - position[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist <= radius:
                results.append((rubble, dist))

        # Sort by distance
        results.sort(key=lambda x: x[1])
        return [r for r, _ in results]

    def get_entity_observations(self) -> Tuple[List[float], List[float]]:
        """
        Get the observations for all entities, separated by type

        Returns:
            Tuple of (casualty observations, rubble observations)
        """
        casualty_observations = []
        for casualty in self.casualties:
            casualty_observations.extend(casualty.get_observation())

        rubble_observations = []
        for rubble in self.rubble:
            rubble_observations.extend(rubble.get_observation())

        return casualty_observations, rubble_observations

    def get_combined_observations(self) -> np.ndarray:
        """
        Get the combined observation vector of all entities

        Returns:
            numpy.ndarray: Combined observation vector of casualties and rubble
        """
        casualty_obs, rubble_obs = self.get_entity_observations()
        return np.array(casualty_obs + rubble_obs, dtype=np.float32)

    def count_rescued_casualties(self) -> int:
        """Count how many casualties have been rescued"""
        return sum(1 for c in self.casualties if c.rescued)

    def count_alive_casualties(self) -> int:
        """Count how many casualties are alive but not rescued"""
        return sum(1 for c in self.casualties if c.is_alive and not c.rescued)

    def count_dead_casualties(self) -> int:
        """Count how many casualties are dead"""
        return sum(1 for c in self.casualties if not c.is_alive)

    def count_cleared_rubble(self) -> int:
        """Count how many rubble obstacles have been cleared"""
        return sum(1 for r in self.rubble if r.cleared)

    def get_currently_clearing_rubble(self) -> Optional[Any]:
        """Get the rubble currently being cleared, if any"""
        for rubble in self.rubble:
            if rubble.being_cleared:
                return rubble
        return None

    def get_entity_stats(self) -> Dict:
        """
        Get summary statistics about all entities

        Returns:
            Dict containing counts and states of entities
        """
        return {
            'casualties_total': len(self.casualties),
            'casualties_rescued': self.count_rescued_casualties(),
            'casualties_alive': self.count_alive_casualties(),
            'casualties_dead': self.count_dead_casualties(),
            'rubble_total': len(self.rubble),
            'rubble_cleared': self.count_cleared_rubble(),
            'currently_clearing': 1 if self.get_currently_clearing_rubble() else 0
        }

    def is_casualty_blocked_by_rubble(self, casualty) -> bool:
        """
        Check if a casualty is blocked by any uncleared rubble

        Args:
            casualty: The casualty to check

        Returns:
            True if the casualty is blocked, False otherwise
        """
        casualty_pos = (casualty.body.position[0], casualty.body.position[1])

        for rubble in self.rubble:
            if not rubble.cleared and rubble.is_position_blocked(casualty_pos):
                return True

        return False

    def get_unblocked_casualties(self) -> List:
        """
        Get all casualties that are not blocked by rubble

        Returns:
            List of unblocked casualties
        """
        return [c for c in self.casualties if not self.is_casualty_blocked_by_rubble(c)]

    def get_blocked_casualties(self) -> List:
        """
        Get all casualties that are blocked by rubble

        Returns:
            List of blocked casualties
        """
        return [c for c in self.casualties if self.is_casualty_blocked_by_rubble(c)]