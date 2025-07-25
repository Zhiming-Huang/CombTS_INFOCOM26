import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from environments.qurinet_environment import QurinetEnvironment
from bandits.bg_cts import BGCTS

if __name__ == "__main__":
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        pre_generate_availability=True,
        pre_generate_rewards=True,
        num_rounds=10
    )
    print("All arm mean rewards:")
    print([float(f"{x:.4f}") for x in env.arm_means])
    available_arms = set(range(env.num_arms))
    feas = env.get_feasible_combinations(available_arms)
    print("All feasible path mean rewards:")
    for i, comb in enumerate(feas):
        mean_reward = sum(env.arm_means[arm_id] for arm_id in comb)
        print(f"  Path {i+1}: {env.get_path_info(comb)['path']}, mean reward: {mean_reward:.4f}")
    algo = BGCTS(environment=env, rnd_generator=np.random.default_rng(42))
    total_regret = 0
    print(f"Source: {env.source}, Destination: {env.destination}")
    for round_num in range(env.num_rounds):
        selected_comb = algo.select_combination(round_num)
        rewards = env.get_reward_for_round(selected_comb, round_num)
        mean_selected = sum(env.arm_means[arm_id] for arm_id in selected_comb)
        # Add debug output
        print(f"Round {round_num+1}:")
        print(f"  selected_comb (raw): {selected_comb}")
        print(f"  selected_comb length: {len(selected_comb)}")
        print(f"  unique arms: {len(set(selected_comb))}, has duplicates: {len(selected_comb) != len(set(selected_comb))}")
        print(f"  arm ids: {list(selected_comb)}")
        algo.update_posterior(selected_comb, rewards, round_num)
        available_arms = env.get_available_arms_for_round(round_num)
        optimal_comb = env.get_optimal_combination(available_arms)
        mean_optimal = sum(env.arm_means[arm_id] for arm_id in optimal_comb) if optimal_comb else 0
        regret = mean_optimal - mean_selected
        total_regret += regret
        print(f"  Selected path: {env.get_path_info(selected_comb)['path']}, mean reward: {mean_selected:.4f}")
        print(f"  Optimal path:  {env.get_path_info(optimal_comb)['path']}, mean reward: {mean_optimal:.4f}")
        print(f"  Regret: {regret:.4f}, Cumulative regret: {total_regret:.4f}\n") 