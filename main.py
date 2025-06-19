import argparse
import subprocess
import sys
from experiments import *
import os

def main():
    parser = argparse.ArgumentParser(description="Run a specific experiment script.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        choices=["cifar10", "omniglot", "pcam", "rot_mnist", "onlyGlots", "toyDataset"],
        default="toyDataset",
        help="Which experiment to run."
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        required=False,
        default=False,
        help="Whether to visualize the train and validation losses"
    )
    args = parser.parse_args()

    script_name = f"experiments/run_{args.experiment}.py"
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    try:
        if not args.visualize:
            subprocess.run([sys.executable, script_path, "-m"], check=True)
        else:
            subprocess.run([sys.executable, script_path, "-m", "-v"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment '{script_name}' failed with exit code {e.returncode}.")
    except Exception as e:
        print(f"Error running '{script_name}': {e}")

if __name__ == "__main__":
    main()