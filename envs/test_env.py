import time
import numpy as np
from disaster_env import DisasterResponseEnv
import pygame
import argparse


def main(static_mode=False):
    """
    Run the disaster environment test

    Args:
        static_mode: If True, only render the environment without simulation
    """
    # Create environment with rendering enabled
    env = DisasterResponseEnv(render_mode='human', world_size=(800, 600))

    # Reset environment
    observation, info = env.reset()

    # Example of how to get casualty and rubble observations directly from entity_manager
    def print_entity_observations():
        # Get casualty observations
        print("Casualty observations:")
        for i, casualty in enumerate(env.entity_manager.get_casualties()):
            casualty_obs = casualty.get_observation()
            print(f"  Casualty {i}: Position=({casualty_obs[0]:.2f}, {casualty_obs[1]:.2f}), "
                  f"Rescued={bool(casualty_obs[2])}, Severity={casualty_obs[3]}, "
                  f"Reward={casualty_obs[4]:.2f}, Alive={bool(casualty_obs[5])}")

        # Get rubble observations
        print("Rubble observations:")
        for i, rubble in enumerate(env.entity_manager.get_rubble()):
            rubble_obs = rubble.get_observation()
            print(f"  Rubble {i}: Position=({rubble_obs[0]:.2f}, {rubble_obs[1]:.2f}), "
                  f"Size=({rubble_obs[2]:.2f}, {rubble_obs[3]:.2f}), "
                  f"Cleared={bool(rubble_obs[4])}, Progress={rubble_obs[5]:.2f}")

        # Example of using entity manager for spatial queries
        agent_pos = (env.agent.position[0], env.agent.position[1])
        print(f"\nAgent position: {agent_pos}")

        # Find nearest casualty
        nearest_casualty, dist = env.entity_manager.find_nearest_casualty(agent_pos)
        if nearest_casualty:
            print(f"Nearest casualty: distance={dist:.2f}, severity={nearest_casualty.severity}")

        # Find nearest rubble
        nearest_rubble, dist = env.entity_manager.find_nearest_rubble(agent_pos)
        if nearest_rubble:
            print(f"Nearest rubble: distance={dist:.2f}, cleared={nearest_rubble.cleared}")

    # Print initial observations for demonstration of retrieving casualty and rubble data
    print_entity_observations()
    # Comment out for random walk simulation.
    static_mode = True
    env.render()
    if static_mode:
        print("\nEnvironment created and reset. Displaying static environment.")
        print("Press R to reset, ESC to quit.")

        # Keep the window open until user closes it
        try:
            running = True
            while running:
                # Keep rendering the environment
                env.render()
                time.sleep(0.1)  # Small delay to prevent using 100% CPU

        except KeyboardInterrupt:
            print("Display stopped by user.")
    else:
        print("\nEnvironment created and reset. Starting simple random agent test...")

        # Run a simple simulation with random actions
        terminated = truncated = False
        total_reward = 0
        max_steps = 1000  # Limit steps for testing

        for step in range(max_steps):
            if terminated or truncated:
                break

            # Take a semi-random action (with some bias toward moving forward)
            # Calculate agent position
            agent_position = (observation[0], observation[1])  # x, y from observation

            # Using entity manager for more efficient querying (showing both approaches)
            # 1. Old approach of scanning through observations
            nearby_casualty = False
            casualty_start_idx = 5  # Agent features end at index 4

            # Loop through all casualties in the observation
            for i in range(len(env.entity_manager.get_casualties())):
                # Each casualty has 6 features, starting after agent features
                casualty_idx = casualty_start_idx + (i * 6)
                casualty_x = observation[casualty_idx]
                casualty_y = observation[casualty_idx + 1]
                is_rescued = observation[casualty_idx + 2] > 0.5
                is_alive = observation[casualty_idx + 5] > 0.5

                # Calculate distance to this casualty
                dx = agent_position[0] - casualty_x
                dy = agent_position[1] - casualty_y
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # Check if this casualty is nearby, alive, and not rescued
                if distance <= 2.0 and not is_rescued and is_alive:
                    nearby_casualty = True
                    break

            # 2. New approach using entity manager (more efficient, but not used here)
            # nearby_casualties = env.entity_manager.get_casualties_in_radius(
            #     agent_position, 2.0, only_alive=True, only_unrescued=True
            # )
            # nearby_casualty = len(nearby_casualties) > 0

            # Check for nearby rubble to clear (using original observation scanning)
            nearby_rubble = False
            agent_features = 5
            casualty_features = 6
            rubble_features = 6

            num_casualties = len(env.entity_manager.get_casualties())
            rubble_start_idx = agent_features + (casualty_features * num_casualties)

            for i in range(len(env.entity_manager.get_rubble())):
                rubble_idx = rubble_start_idx + (i * rubble_features)
                rubble_x = observation[rubble_idx]
                rubble_y = observation[rubble_idx + 1]
                is_cleared = observation[rubble_idx + 4] > 0.5

                # Calculate distance to this rubble
                dx = agent_position[0] - rubble_x
                dy = agent_position[1] - rubble_y
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # Check if rubble is nearby and not cleared
                if distance <= 2.0 and not is_cleared:
                    nearby_rubble = True
                    break

            # Take action based on proximity
            if nearby_casualty:
                # If near a viable casualty, attempt rescue
                action = np.array([0.0, 0.0, 1.0, 0.0])  # Try to rescue, no clearing
            elif nearby_rubble:
                # If near uncleared rubble, attempt to clear it
                action = np.array([0.0, 0.0, 0.0, 1.0])  # Try to clear rubble, no rescue
            elif np.random.random() < 0.7:
                # Otherwise, explore with movement 70% of time
                action = np.array([
                    np.random.uniform(-0.5, 0.5),  # Steering
                    np.random.uniform(0.2, 1.0),  # Forward acceleration
                    0.0,  # No rescue attempt
                    0.0  # No rubble clearing attempt
                ])
            else:
                # Random movement the rest of the time
                action = np.array([
                    np.random.uniform(-1.0, 1.0),  # Full range steering
                    np.random.uniform(-0.3, 1.0),  # Mostly forward but some backward
                    0.0,  # No rescue attempt
                    0.0  # No rubble clearing attempt
                ])

            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            # Print info every 10 steps
            if step % 10 == 0:
                print(f"Step {step}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
                print(f"Info: {info}")
                print("-" * 50)

            # Small pause for visualization
            time.sleep(0.01)

        print(f"Test complete after {step + 1} steps. Total reward: {total_reward:.2f}")

        # Keep the window open until user closes it
        if not (terminated or truncated):
            print("Simulation ended. Close the window to exit.")
            try:
                running = True
                while running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Simulation stopped by user.")

    # Close environment
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run disaster response environment test')
    parser.add_argument('--static', action='store_true', help='Run in static mode without simulation')
    args = parser.parse_args()

    main(static_mode=args.static)