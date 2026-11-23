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


def get_all_gestures():
    """Returns a dictionary of all gestures with DB name and image counts."""
    ensure_directories()
    db_gestures = {}
    if DATABASE_PATH.exists():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT g_id, g_name FROM gesture ORDER BY g_id ASC")
                for row in cursor.fetchall():
                    db_gestures[int(row[0])] = row[1]
        except Exception as e:
            print(f"[ERROR] Database error: {e}")

    folder_gestures = {}
    if GESTURES_DIR.exists():
        for item in GESTURES_DIR.iterdir():
            if item.is_dir() and item.name.isdigit():
                g_id = int(item.name)
                num_imgs = len(list(item.glob("*.jpg")))
                folder_gestures[g_id] = num_imgs

    all_ids = sorted(set(list(db_gestures.keys()) + list(folder_gestures.keys())))
    return all_ids, db_gestures, folder_gestures


def print_gesture_table():
    """Prints a cleanly formatted table of all gestures."""
    all_ids, db_gestures, folder_gestures = get_all_gestures()

    print("\n" + "=" * 68)
    print("                CURRENT GESTURE DATABASE                     ")
    print("=" * 68)
    print(f"{'ID':<6} | {'Gesture Name':<32} | {'Images':<8} | {'Status':<14}")
    print("-" * 68)

    if not all_ids:
        print("  (Database is currently empty. No gestures found.)")
        print("=" * 68)
        print("-> Next recommended ID to use for new gesture: 1")
        return all_ids, db_gestures, folder_gestures

    total_images = 0
    for gid in all_ids:
        name = db_gestures.get(gid, "(Not in DB)")
        img_count = folder_gestures.get(gid, 0)
        total_images += img_count

        if gid in db_gestures and img_count > 0:
            status = "Ready"
        elif gid in db_gestures and img_count == 0:
            status = "No images"
        elif gid not in db_gestures and img_count > 0:
            status = "Folder only"
        else:
            status = "Empty"

        print(f"{gid:<6} | {name:<32} | {img_count:<8} | {status:<14}")

    print("=" * 68)
    print(f"Total Gestures: {len(all_ids)}  |  Total Images: {total_images}")
    next_id = max(all_ids) + 1 if all_ids else 1
    print(f"Next recommended ID: {next_id}")
    print("=" * 68)

    return all_ids, db_gestures, folder_gestures


def delete_gesture_by_id(g_id: int):
    """Deletes gesture images and database record for the given ID."""
    folder_path = GESTURES_DIR / str(g_id)
    deleted_images = 0

    # 1. Delete image folder
    if folder_path.exists() and folder_path.is_dir():
        deleted_images = len(list(folder_path.glob("*.jpg")))
        shutil.rmtree(folder_path)

    # 2. Delete database entry
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
        except Exception as e:
            print(f"[ERROR] Database error: {e}")

    if deleted_images > 0 or removed_from_db:
        print(f"\n>>> [SUCCESS] Deleted Gesture ID {g_id} ('{g_name or 'Unknown'}') - {deleted_images} images removed from disk and database.")
        print(">>> Tip: Remember to run 'load_images.py' & 'cnn_model_train.py' when done updating gestures.")
    else:
        print(f"\n>>> [WARNING] Gesture ID {g_id} not found in database or folders.")


def delete_all_data():
    """Deletes all gesture images, resets DB, and clears datasets."""
    deleted_folders = 0
    if GESTURES_DIR.exists():
        for item in GESTURES_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                deleted_folders += 1
            elif item.is_file():
                item.unlink()

    if DATABASE_PATH.exists():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                conn.execute("DROP TABLE IF EXISTS gesture")
                conn.execute(
                    "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
                )
                conn.commit()
        except Exception as e:
            print(f"[ERROR] Database reset error: {e}")

    if DATASETS_DIR.exists():
        for item in DATASETS_DIR.iterdir():
            if item.is_file():
                item.unlink()

    print(f"\n>>> [SUCCESS] All old data deleted! ({deleted_folders} classes removed, database reset).")
    print(">>> You now have a clean slate to capture brand new gestures starting from ID 1.")


def run_manager():
    """Interactive loop for viewing and deleting gestures with minimal typing."""
    while True:
        all_ids, db_gestures, folder_gestures = print_gesture_table()

        print("\nOPTIONS:")
        print("  - Type an ID number (e.g. 500) to delete that gesture")
        print("  - Type 'all' to delete ALL old data (clean slate)")
        print("  - Type 'r' to refresh the list")
        print("  - Type 'q' to quit")
        print("-" * 68)

        choice = input("Enter choice: ").strip()

        if choice.lower() in ("q", "quit", "exit"):
            print("Exiting Gesture Data Manager. Goodbye!")
            break

        if choice.lower() in ("r", "refresh"):
            continue

        if choice.lower() == "all":
            confirm = input("\n[WARNING] Are you sure you want to delete ALL gestures from disk and database? Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                delete_all_data()
            else:
                print("Action cancelled.")
            input("\nPress Enter to continue...")
            continue

        if choice.isdigit():
            target_id = int(choice)
            name = db_gestures.get(target_id, "Unknown")
            imgs = folder_gestures.get(target_id, 0)

            print(f"\nTarget Selected: ID {target_id} | Name: '{name}' | Images: {imgs}")
            confirm = input(f"Are you sure you want to delete ID {target_id} ('{name}')? (y/n): ").strip().lower()

            if confirm in ("y", "yes"):
                delete_gesture_by_id(target_id)
            else:
                print("Deletion cancelled.")

            input("\nPress Enter to continue...")
        else:
            print("[ERROR] Please enter a valid ID number, 'all', 'r', or 'q'.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    run_manager()
