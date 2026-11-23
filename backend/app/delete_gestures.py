import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Add backend/app to path if needed for paths import
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from paths import (
    DATABASE_PATH,
    DATASETS_DIR,
    GESTURES_DIR,
    MODEL_PATH,
    ensure_directories,
)


def delete_single_gesture(g_id: int):
    """Deletes a specific gesture by its ID from disk and database."""
    ensure_directories()
    folder_path = GESTURES_DIR / str(g_id)
    deleted_images = 0

    # 1. Remove image folder
    if folder_path.exists() and folder_path.is_dir():
        deleted_images = len(list(folder_path.glob("*.jpg")))
        shutil.rmtree(folder_path)
        print(f"[OK] Deleted image folder for ID {g_id} ({deleted_images} images removed).")
    else:
        print(f"[INFO] No image folder found for ID {g_id}.")

    # 2. Remove from database
    removed_from_db = False
    g_name = ""
    if DATABASE_PATH.exists():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT g_name FROM gesture WHERE g_id = ?", (g_id,))
                row = cursor.fetchone()
                if row:
                    g_name = row[0]
                    cursor.execute("DELETE FROM gesture WHERE g_id = ?", (g_id,))
                    conn.commit()
                    removed_from_db = True
                    print(f"[OK] Removed gesture '{g_name}' (ID {g_id}) from database.")
        except Exception as e:
            print(f"[ERROR] Database error: {e}")

    if not folder_path.exists() and not removed_from_db:
        print(f"[WARNING] Gesture ID {g_id} was not found in folders or database.")
    else:
        print(f"[SUCCESS] Gesture ID {g_id} ({g_name or 'Unknown'}) deleted completely.")
        print("Note: If you already packaged datasets, please run 'load_images.py' and 'cnn_model_train.py' again.")


def delete_all_data(include_model: bool = False):
    """Deletes all gestures, resets database, and clears datasets to start completely fresh."""
    ensure_directories()

    # 1. Clear gestures folder
    deleted_folders = 0
    if GESTURES_DIR.exists():
        for item in GESTURES_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                deleted_folders += 1
            elif item.is_file():
                item.unlink()
        print(f"[OK] Cleared all gesture folders ({deleted_folders} gesture classes deleted).")

    # 2. Reset database table
    if DATABASE_PATH.exists():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute("DROP TABLE IF EXISTS gesture")
                conn.execute(
                    "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
                )
                conn.commit()
            print("[OK] Reset gesture database table (all entries cleared).")
        except Exception as e:
            print(f"[ERROR] Database reset error: {e}")

    # 3. Clear packaged datasets
    if DATASETS_DIR.exists():
        for item in DATASETS_DIR.iterdir():
            if item.is_file():
                item.unlink()
        print("[OK] Cleared packaged datasets (train/test/val files removed).")

    # 4. Optionally clear old model
    if include_model and MODEL_PATH.exists():
        MODEL_PATH.unlink()
        print("[OK] Removed old trained model file.")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL OLD DATA HAS BEEN CLEARED!")
    print("You now have a clean slate to train brand new gestures.")
    print("Start capturing gestures with ID 1 using 'create_gestures.py'.")
    print("=" * 60)


def interactive_menu():
    print("=" * 60)
    print("       SIGN LANGUAGE DATA MANAGEMENT & CLEANUP TOOL       ")
    print("=" * 60)
    print("1. View all gestures (List IDs and Names)")
    print("2. Delete a specific gesture by ID (e.g. 500)")
    print("3. DELETE ALL OLD DATA (Fresh restart for new gestures)")
    print("4. Exit")
    print("=" * 60)

    choice = input("Enter choice (1/2/3/4): ").strip()

    if choice == "1":
        from list_gestures import list_all_gestures
        list_all_gestures()
    elif choice == "2":
        gid_str = input("Enter Gesture ID to delete (e.g. 500): ").strip()
        if gid_str.isdigit():
            delete_single_gesture(int(gid_str))
        else:
            print("[ERROR] Invalid ID. Please enter a number.")
    elif choice == "3":
        confirm = input(
            "WARNING: This will permanently delete ALL gesture images, database records, and datasets!\nAre you sure? Type 'yes' to confirm: "
        ).strip().lower()
        if confirm == "yes":
            del_model = input("Do you also want to delete the old trained model file (.h5)? (y/n): ").strip().lower() == "y"
            delete_all_data(include_model=del_model)
        else:
            print("[CANCELLED] Operation cancelled. No data was deleted.")
    else:
        print("Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Data Cleanup Tool")
    parser.add_argument("--id", type=int, help="Delete a specific gesture ID")
    parser.add_argument("--all", action="store_true", help="Delete ALL old gestures and start fresh")
    parser.add_argument("--with-model", action="store_true", help="When using --all, also delete the old model")
    args = parser.parse_args()

    if args.id is not None:
        delete_single_gesture(args.id)
    elif args.all:
        delete_all_data(include_model=args.with_model)
    else:
        interactive_menu()
