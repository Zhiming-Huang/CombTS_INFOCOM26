import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from environments.qurinet_environment import QurinetEnvironment

if __name__ == "__main__":
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        pre_generate_availability=True,
        pre_generate_rewards=True,
        num_rounds=1
    )
    env.visualize_network() 