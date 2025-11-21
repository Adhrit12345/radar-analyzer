# List output files and print summaries of insights.md and confusion_matrices.json
import os, json
out_dir = "C:/Users/ADHRIT/rvce/ai-project/training-output"
files = sorted(os.listdir(out_dir))
print("Files in out dir:", files)

# print first 4000 chars of insights.md
insights_path = os.path.join(out_dir, "insights.md")
if os.path.exists(insights_path):
    with open(insights_path, "r") as f:
        s = f.read()
    print("\n--- insights.md (first 4000 chars) ---\n")
    print(s[:4000])
else:
    print("insights.md not found")

# print confusion_matrices.json
cm_path = os.path.join(out_dir, "confusion_matrices.json")
if os.path.exists(cm_path):
    with open(cm_path, "r") as f:
        cm = json.load(f)
    print("\n--- confusion_matrices.json ---\n")
    print(json.dumps(cm, indent=2))
else:
    print("confusion_matrices.json not found")

# show a small sample of positions.json
pos_path = os.path.join(out_dir, "positions.json")
if os.path.exists(pos_path):
    with open(pos_path, "r") as f:
        positions = json.load(f)
    keys = list(positions.keys())[:10]
    print("\n--- positions.json keys (first 10) and one sample entry ---\n")
    print(keys)
    if keys:
        print(json.dumps(positions[keys[0]][:5], indent=2))
else:
    print("positions.json not found")
