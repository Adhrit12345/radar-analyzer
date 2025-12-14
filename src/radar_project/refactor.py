from pathlib import Path
import json

# Directory containing output files
out_dir = Path("C:/Users/ADHRIT/rvce/ai-project/prediction-output")

# List files
files = sorted(out_dir.iterdir())
print("📁 Files in output directory:")
for file in files:
    print("  -", file.name)

# Display first 4000 characters of predictions.md
predictions_path = out_dir / "predictions.md"
if predictions_path.exists():
    print("\n📄 --- predictions.md (first 4000 chars) ---\n")
    content = predictions_path.read_text()[:4000]
    print(content)
else:
    print("❌ predictions.md not found")

# Removed pet vs human section completely from code

# Show a small sample of positions_pred.json
positions_path = out_dir / "positions_pred.json"
if positions_path.exists():
    positions = json.loads(positions_path.read_text())
    keys = list(positions.keys())[:10]
    print("\n📍 --- positions_pred.json keys (first 10) & sample entry ---\n")
    print(keys)
    if keys:
        example = positions[keys[0]][:5] if isinstance(positions[keys[0]], list) else positions[keys[0]]
        print(json.dumps(example, indent=2))
else:
    print("❌ positions_pred.json not found")
