#!/usr/bin/env python3
"""Add full-setting T-CL-SG curves to the saved UCSB results file."""

from pathlib import Path
import multiprocessing as mp
import pickle
import sys
import time

import numpy as np
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from example.example_ucsb_comprehensive_parallel import (  # noqa: E402
    plot_from_saved_data,
    save_simulation_results,
    setup_ucsb_environment,
    run_single_algorithm,
)
from src.bandits.truncated_cl_sg import TCLSG  # noqa: E402


def build_tclsg_configs(env_config, num_rounds, num_runs, main_seed, gamma_values):
    """Use the same expanded algorithm order as the full experiment runner."""
    total_algorithms = 3 + 3 * len(gamma_values)
    main_rng = np.random.default_rng(main_seed)
    all_generators = main_rng.spawn(total_algorithms * (1 + num_runs))

    # Order: 3 base algorithms, 4 CTS-G, 4 CL-SG, then 4 T-CL-SG.
    generator_idx = (3 + 2 * len(gamma_values)) * (1 + num_runs)
    configs = []
    for gamma in gamma_values:
        alg_key = f"t-cl-sg_gamma_{gamma}"
        env_generator = all_generators[generator_idx]
        generator_idx += 1
        alg_generators = list(all_generators[generator_idx:generator_idx + num_runs])
        generator_idx += num_runs
        configs.append((alg_key, TCLSG, env_config, num_rounds, num_runs, env_generator, alg_generators))
    return configs


def main():
    num_rounds = 10000
    num_runs = 100
    main_seed = 123
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    source = "10.1.1.100"
    destination = "10.1.1.102"
    trace_period = "1144393236-1144450070"
    max_path_length = 3
    pkl_path = PROJECT_ROOT / "output/data/ucsb_results_r10000_n100_s123_10_1_1_100_to_10_1_1_102_1144393236-1144450070.pkl"

    with pkl_path.open("rb") as f:
        saved = pickle.load(f)
    results = saved["results"]
    metadata = saved["metadata"]

    missing_gammas = [gamma for gamma in gamma_values if f"t-cl-sg_gamma_{gamma}" not in results]
    print(f"Missing T-CL-SG full curves: {missing_gammas}")
    if missing_gammas:
        env_config = setup_ucsb_environment(
            num_rounds,
            fixed_source=source,
            fixed_destination=destination,
            trace_period=trace_period,
            max_path_length=max_path_length,
        )
        configs = [
            config for config in build_tclsg_configs(env_config, num_rounds, num_runs, main_seed, gamma_values)
            if float(config[0].split("_gamma_")[1]) in missing_gammas
        ]

        print(f"Running {len(configs)} T-CL-SG full configurations with {num_runs} runs each.")
        start = time.time()
        with mp.Pool(processes=min(mp.cpu_count(), len(configs))) as pool:
            results_list = list(tqdm(pool.imap(run_single_algorithm, configs), total=len(configs), desc="T-CL-SG"))
        print(f"T-CL-SG completion time: {time.time() - start:.2f}s")

        for alg_name, regrets_array, rewards_array in results_list:
            avg_cumulative_regrets = np.mean(regrets_array, axis=0)
            std_cumulative_regrets = np.std(regrets_array, axis=0)
            confidence_interval = stats.t.interval(
                0.95,
                num_runs - 1,
                loc=avg_cumulative_regrets,
                scale=std_cumulative_regrets / np.sqrt(num_runs),
            )
            results[alg_name] = {
                "avg_cumulative_regrets": avg_cumulative_regrets,
                "std_cumulative_regrets": std_cumulative_regrets,
                "confidence_interval": confidence_interval,
                "all_regrets": regrets_array,
                "avg_cumulative_rewards": np.mean(rewards_array, axis=0),
                "std_cumulative_rewards": np.std(rewards_array, axis=0),
                "final_regret": avg_cumulative_regrets[-1],
                "final_regret_std": std_cumulative_regrets[-1],
                "final_regret_ci": (confidence_interval[0][-1], confidence_interval[1][-1]),
            }
            print(f"{alg_name}: final regret {avg_cumulative_regrets[-1]:.2f} +/- {std_cumulative_regrets[-1]:.2f}")

        metadata["includes_t_cl_sg_full"] = True
        metadata["t_cl_sg_full_source"] = "scripts/run_tclsg_full_ucsb.py"
        save_simulation_results(results, str(pkl_path), metadata)

    changed = False
    for gamma in gamma_values:
        alg_key = f"t-cl-sg_gamma_{gamma}"
        alg_data = results.get(alg_key)
        if alg_data and "final_regret" not in alg_data:
            ci = alg_data["confidence_interval"]
            alg_data["final_regret"] = alg_data["avg_cumulative_regrets"][-1]
            alg_data["final_regret_ci"] = (ci[0][-1], ci[1][-1])
            changed = True
    if changed:
        save_simulation_results(results, str(pkl_path), metadata)

    plot_from_saved_data(str(pkl_path), output_dir=str(PROJECT_ROOT / "output/images"))


if __name__ == "__main__":
    main()
