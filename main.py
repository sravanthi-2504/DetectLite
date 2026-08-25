import argparse
import subprocess
import sys


def run(cmd):
    print("\n>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="MobileViT Object Detection")
    parser.add_argument("command", choices=[
        "test", "train", "evaluate", "profile", "predict", "webcam"
    ])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    a = parser.parse_args()

    mapping = {
        "test": ["scripts/test_model.py"],
        "train": ["training/train.py"],
        "evaluate": ["training/evaluate.py"],
        "profile": ["scripts/profile.py"],
        "predict": ["inference/predict.py"],
        "webcam": ["inference/webcam.py"],
    }
    run([sys.executable] + mapping[a.command] + a.args)


if __name__ == "__main__":
    main()
