import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from list_gestures import list_all_gestures

PYTHON_EXE = sys.executable


def run_script(script_name: str, step_desc: str) -> bool:
    script_path = CURRENT_DIR / script_name
    print("\n" + "=" * 65)
    print(f">> {step_desc}")
    print(f">> Running: {script_name}")
    print("=" * 65)

    try:
        result = subprocess.run(
            [PYTHON_EXE, str(script_path)],
            cwd=str(CURRENT_DIR),
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}: {script_name}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error running {script_name}: {e}")
        return False


def run_three_training_steps() -> bool:
    """Executes the 3 training commands in sequence:
       1. Rotate_images.py (Augment / flip images for left & right hands)
       2. load_images.py (Package and split train/test/val datasets)
       3. cnn_model_train.py (Train the CNN deep learning model)
    """
    print("\n" + "=" * 65)
    print("       STARTING 3-STEP TRAINING PIPELINE (ONE-CLICK)       ")
    print("=" * 65)

    # Step 1: Augment images
    ok = run_script("Rotate_images.py", "Step 1 of 3: Augmenting Images (Flip Left/Right)")
    if not ok:
        print("\n[STOPPED] Augmentation failed. Aborting remaining training steps.")
        return False

    # Step 2: Package datasets
    ok = run_script("load_images.py", "Step 2 of 3: Packaging & Splitting Datasets")
    if not ok:
        print("\n[STOPPED] Dataset packaging failed. Aborting model training.")
        return False

    # Step 3: Train CNN model
    ok = run_script("cnn_model_train.py", "Step 3 of 3: Training the CNN Neural Network")
    if not ok:
        print("\n[STOPPED] Model training failed.")
        return False

    print("\n" + "=" * 65)
    print("       >>> ALL 3 TRAINING STEPS COMPLETED SUCCESSFULLY! <<<")
    print("The trained model has been saved to 'backend/models/cnn_model_keras2.h5'")
    print("Your new gestures are now active in the web and desktop recognition app!")
    print("=" * 65)
    return True


def capture_new_gesture() -> bool:
    """Runs create_gestures.py to record a new gesture ID and name with camera."""
    print("\n" + "=" * 65)
    print("                  CURRENT GESTURES LIST                     ")
    print("=" * 65)
    list_all_gestures()

    print("\nStarting Gesture Image Capture...")
    print("Instructions:")
    print("  1. Enter the new Gesture ID and Name when prompted.")
    print("  2. In the camera window, put hand(s) inside the green box.")
    print("  3. Press 'c' to start capturing images (captures 300 pictures).")
    print("  4. Press 'q' if you want to quit early.")
    print("-" * 65)

    return run_script("create_gestures.py", "Capture New Gesture Images (1 or 2 Hands)")


def full_workflow():
    """Captures a new gesture, then automatically executes the 3 training steps."""
    capture_ok = capture_new_gesture()
    if not capture_ok:
        print("\n[INFO] Gesture capture did not complete. Training cancelled.")
        return

    print("\n" + "-" * 65)
    choice = input("Capture finished! Train the model with the 3 steps now? (y/n, default 'y'): ").strip().lower()
    if choice in ("", "y", "yes"):
        run_three_training_steps()
    else:
        print("\n[SAVED] New gesture images are saved. You can train the model later anytime.")


def main():
    while True:
        print("\n" + "=" * 65)
        print("        SIGN LANGUAGE - GESTURE CAPTURE & TRAIN TOOL        ")
        print("=" * 65)
        print("  1. [Full Workflow] Capture New Gesture + Auto-Train (All Steps)")
        print("  2. [3 Training Steps Only] Train / Retrain Model (Rotate -> Load -> Train)")
        print("  3. Capture New Gesture Only (without training)")
        print("  4. View All Current Gestures & IDs")
        print("  5. Exit")
        print("=" * 65)

        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            full_workflow()
            input("\nPress Enter to return to menu...")
        elif choice == "2":
            run_three_training_steps()
            input("\nPress Enter to return to menu...")
        elif choice == "3":
            capture_new_gesture()
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            list_all_gestures()
            input("\nPress Enter to return to menu...")
        elif choice in ("5", "q", "quit", "exit"):
            print("Exiting tool. Goodbye!")
            break
        else:
            print("[ERROR] Invalid choice. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()
