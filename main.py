#!/usr/bin/env python
"""
main.py
=======
Entry point for the Yoga Pose Detection and Correction project.

Orchestrates the complete pipeline:
    1. Data cleaning (remove corrupted/duplicate images)
    2. Dataset statistics
    3. Model training (EfficientNetB0 + MobileNetV2)
    4. Model evaluation
    5. Launch Streamlit app

Usage:
    python main.py                  # show menu
    python main.py --train          # run full training pipeline
    python main.py --evaluate       # evaluate trained models
    python main.py --clean          # clean corrupted/duplicate images
    python main.py --app            # launch Streamlit web app
    python main.py --all            # run everything (clean → train → evaluate)
"""

import os
import sys
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import config


def print_banner():
    banner = """
    =====================================================
       YOGA POSE DETECTION & CORRECTION
       Deep Learning | Computer Vision | Streamlit
    =====================================================
    """
    print(banner)
    config.print_config()


def cmd_clean():
    """Clean corrupted and duplicate images from the dataset."""
    print("\n  RUNNING: Data cleaning pipeline")
    print("=" * 60)
    from data_processing import clean_dataset
    summary = clean_dataset()
    print(f"\n  Summary:")
    print(f"    Corrupted removed : {summary['corrupted_removed']}")
    print(f"    Duplicates removed: {summary['duplicates_removed']}")
    print(f"    Total removed     : {summary['corrupted_removed'] + summary['duplicates_removed']}")
    return summary


def cmd_train():
    """Run the full training pipeline."""
    print("\n  RUNNING: Model training pipeline")
    print("=" * 60)
    from model_train import train
    results = train()
    return results


def cmd_evaluate():
    """Evaluate trained models on the test set."""
    print("\n  RUNNING: Model evaluation")
    print("=" * 60)
    from evaluate import main as eval_main
    # Redirect args
    sys.argv = [sys.argv[0], "--model", "all"]
    try:
        eval_main()
    except SystemExit:
        pass


def cmd_app():
    """Launch the Streamlit web application."""
    print("\n  RUNNING: Launching Streamlit app...")
    print("=" * 60)
    print(f"  App URL: http://localhost:8501")
    print("=" * 60)
    streamlit_cmd = f'streamlit run "{os.path.join(config.BASE_DIR, "app.py")}"'
    os.system(streamlit_cmd)


def cmd_dataset_stats():
    """Print dataset statistics."""
    print("\n  DATASET STATISTICS")
    print("=" * 60)
    from data_processing import load_dataset_paths
    paths, labels, class_names = load_dataset_paths()
    print(f"  Total images: {len(paths)}")
    print(f"  Total classes: {len(class_names)}")
    print()
    for i, name in enumerate(class_names):
        count = int((labels == i).sum())
        pct = count / len(paths) * 100
        print(f"    {name:<20s} {count:>4d} images ({pct:>5.1f}%)")
    print(f"\n  {'-' * 40}")
    print(f"    {'TOTAL':<20s} {len(paths):>4d} images (100%)")


def main():
    parser = argparse.ArgumentParser(
        description="Yoga Pose Detection and Correction — Main Entry Point",
    )
    parser.add_argument("--all", action="store_true", help="Run full pipeline: clean → train → evaluate")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained models")
    parser.add_argument("--clean", action="store_true", help="Clean corrupted/duplicate images")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web app")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    args = parser.parse_args()

    print_banner()

    # If no args, show menu
    if not any(vars(args).values()):
        print("\n  Available commands:")
        print(f"    python main.py --all         Run full pipeline")
        print(f"    python main.py --train       Train models")
        print(f"    python main.py --evaluate    Evaluate trained models")
        print(f"    python main.py --clean       Clean dataset")
        print(f"    python main.py --app         Launch Streamlit web app")
        print(f"    python main.py --stats       Show dataset statistics")
        print()
        return

    if args.clean:
        cmd_clean()

    if args.stats:
        cmd_dataset_stats()

    if args.train:
        cmd_train()

    if args.evaluate:
        cmd_evaluate()

    if args.all:
        cmd_clean()
        cmd_dataset_stats()
        cmd_train()
        cmd_evaluate()

    if args.app:
        cmd_app()


if __name__ == "__main__":
    main()
