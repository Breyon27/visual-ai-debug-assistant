
import csv
from datetime import datetime

def log_ai_fix(frame_name, prompt, fix_text, output_file="ai_generated_fixes.csv"):
    now = datetime.now().isoformat(timespec="seconds")
    header = ["Frame", "Prompt", "Fix", "Timestamp", "Source"]

    row = [frame_name, prompt, fix_text, now, "GPT-4o"]

    try:
        file_exists = False
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print(f"Logging failed: {e}")
