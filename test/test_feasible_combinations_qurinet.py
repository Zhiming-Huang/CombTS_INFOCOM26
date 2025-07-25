import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from environments.qurinet_environment import QurinetEnvironment

if __name__ == "__main__":
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        pre_generate_availability=True,
        pre_generate_rewards=True,
        num_rounds=10  # Only test first 10 rounds for easier output
    )
    print(f"Source: {env.source}, Destination: {env.destination}")
    for round_num in range(env.num_rounds):
        available_arms = env.get_available_arms_for_round(round_num)
        feas = env.get_feasible_combinations(available_arms)
        print(f"\nRound {round_num+1}: {len(feas)} feasible combinations")
        for i, comb in enumerate(feas):
            path_info = env.get_path_info(comb)
            print(f"  Path {i+1}: {path_info['path']}, reward mean: {path_info['expected_reward']:.3f}") 