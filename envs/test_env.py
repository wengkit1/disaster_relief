import time
import numpy as np
from disaster_env import DisasterResponseEnv
import pygame

def main():
    # Create environment with rendering enabled
    env = DisasterResponseEnv(render_mode='human', world_size=(800, 600),
                              num_casualties=5, num_rubble=10)

    # Reset environment
    observation, info = env.reset()

    print("Environment created and reset. Starting simple random agent test...")

    # Run a simple simulation with random actions
    terminated = truncated = False
    total_reward = 0
    max_steps = 300  # Limit steps for testing

    for step in range(max_steps):
        if terminated or truncated:
            break

        # Take a semi-random action (with some bias toward moving forward)
        if np.random.random() < 0.7:
            # 70% of the time, try to move in some direction
            action = np.array([
                np.random.uniform(-0.5, 0.5),  # Steering (less aggressive)
                np.random.uniform(0.2, 1.0),  # Mostly forward acceleration
                0.0,  # No clearance request
                0.0  # No rescue attempt
            ])
        elif np.random.random() < 0.5:
            # Try to clear rubble
            action = np.array([0.0, 0.0, 1.0, 0.0])
        else:
            # Try to rescue
            action = np.array([0.0, 0.0, 0.0, 1.0])

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
                # Process pygame events to keep the window responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break

                env.render()
                pygame.display.update()  # Ensure display is updated
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Simulation stopped by user.")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()