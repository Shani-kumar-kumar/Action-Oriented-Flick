import os
import sqlite3
import sys
from pathlib import Path

# Add backend/app to path if needed for paths import
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from paths import DATABASE_PATH, GESTURES_DIR, ensure_directories

def list_all_gestures():
    ensure_directories()
    
    # Query database
    db_gestures = {}
    if DATABASE_PATH.exists():
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT g_id, g_name FROM gesture ORDER BY g_id ASC")
                for row in cursor.fetchall():
                    db_gestures[int(row[0])] = row[1]
        except Exception as e:
            print(f"Database error: {e}")

    # Inspect gesture folders
    folder_gestures = {}
    if GESTURES_DIR.exists():
        for item in GESTURES_DIR.iterdir():
            if item.is_dir() and item.name.isdigit():
                g_id = int(item.name)
                # Count .jpg images
                num_imgs = len(list(item.glob("*.jpg")))
                folder_gestures[g_id] = num_imgs

    # Merge all unique IDs
    all_ids = sorted(set(list(db_gestures.keys()) + list(folder_gestures.keys())))

    print("=" * 68)
    print(f"{'ID':<6} | {'Gesture Name':<32} | {'Images':<8} | {'Status':<14}")
    print("=" * 68)

    if not all_ids:
        print("  No gestures found. The dataset is currently empty.")
        print("=" * 68)
        print("\nNext recommended Gesture ID to use: 0 or 1")
        return

    total_images = 0
    valid_gestures = 0

    for gid in all_ids:
        name = db_gestures.get(gid, "(Not in DB)")
        img_count = folder_gestures.get(gid, 0)
        total_images += img_count

        if gid in db_gestures and img_count > 0:
            status = "Ready"
            valid_gestures += 1
        elif gid in db_gestures and img_count == 0:
            status = "No images"
        elif gid not in db_gestures and img_count > 0:
            status = "Folder only"
        else:
            status = "Empty"

        print(f"{gid:<6} | {name:<32} | {img_count:<8} | {status:<14}")

    print("=" * 68)
    print(f"Total Gestures: {len(all_ids)}  |  Total Images: {total_images}")
    
    # Suggest next sequential ID
    next_id = max(all_ids) + 1 if all_ids else 1
    print(f"Next recommended Gesture ID: {next_id}")
    print("=" * 68)

if __name__ == "__main__":
    list_all_gestures()
