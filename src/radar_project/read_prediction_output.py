# List output files and generate HTML report with summary and prediction details
import os, json, argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Read prediction output from a specified directory and generate HTML report')
parser.add_argument('-p', '--path', type=str, default="C:/Users/ADHRIT/rvce/ai-project/prediction-output",
                    help='Path to the prediction output directory (default: C:/Users/ADHRIT/rvce/ai-project/prediction-output)')
args = parser.parse_args()

out_dir = args.path
if not os.path.exists(out_dir):
    print(f"Error: Directory '{out_dir}' does not exist")
    exit(1)

files = sorted(os.listdir(out_dir))
print(f"Files in out dir ({out_dir}):", files)

# Load prediction summary JSON
summary_data = {}
summary_path = os.path.join(out_dir, "prediction_summary.json")
if os.path.exists(summary_path):
    with open(summary_path, "r", encoding='utf-8') as f:
        summary_data = json.load(f)
    print(f"Summary data loaded: {summary_data}")
else:
    print(f"Warning: prediction_summary.json not found at {summary_path}")

# Recalculate person count from predictions.json to ensure accuracy
predictions_json_path = os.path.join(out_dir, "predictions.json")
person_count = 0
total_predictions = 0
files_processed = 0
file_stats = {}  # Track stats per file

if os.path.exists(predictions_json_path):
    try:
        with open(predictions_json_path, "r", encoding='utf-8') as f:
            predictions = json.load(f)
        total_predictions = len(predictions)
        
        # Calculate stats per file
        for prediction in predictions:
            source_file = prediction.get('source_file', 'unknown')
            if source_file not in file_stats:
                file_stats[source_file] = {'total': 0, 'persons': 0}
            
            file_stats[source_file]['total'] += 1
            
            if prediction.get('object_pred') == 'person':
                person_count += 1
                file_stats[source_file]['persons'] += 1
        
        files_processed = len(file_stats)
        print(f"Recalculated person count: {person_count} out of {total_predictions} total predictions")
        print(f"Files processed: {files_processed}")
        # Update summary_data with recalculated values
        summary_data['total_person_objects'] = person_count
        summary_data['total_predictions'] = total_predictions
        summary_data['files_processed'] = files_processed
    except Exception as e:
        print(f"Error loading predictions.json: {e}")
        person_count = summary_data.get('total_person_objects', 0)
        total_predictions = summary_data.get('total_predictions', 0)
        files_processed = summary_data.get('files_processed', 0)
else:
    print(f"Warning: predictions.json not found at {predictions_json_path}")
    person_count = summary_data.get('total_person_objects', 0)
    total_predictions = summary_data.get('total_predictions', 0)
    files_processed = summary_data.get('files_processed', 0)

# Load predictions.md for details
predictions_md_path = os.path.join(out_dir, "predictions.md")
predictions_md_content = ""
if os.path.exists(predictions_md_path):
    with open(predictions_md_path, "r", encoding='utf-8') as f:
        predictions_md_content = f.read()

# Generate HTML report
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Output Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px 20px;
            min-height: 100vh;
            color: #2c3e50;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        }}
        h1 {{
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 28px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 35px;
            margin-bottom: 20px;
            border-left: 4px solid #95a5a6;
            padding-left: 15px;
            font-size: 18px;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }}
        th {{
            background-color: #ecf0f1;
            color: #2c3e50;
            padding: 14px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #bdc3c7;
        }}
        td {{
            padding: 14px;
            border-bottom: 1px solid #ecf0f1;
            color: #34495e;
            font-size: 14px;
        }}
        tbody tr {{
            transition: background-color 0.3s ease;
        }}
        tbody tr:hover {{
            background-color: #f8f9fa;
        }}
        tbody tr:last-child td {{
            border-bottom: none;
        }}
        .metric-value {{
            font-weight: 600;
            color: #2980b9;
            font-size: 15px;
        }}
        .details-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #95a5a6;
        }}
        .details-section p {{
            margin: 10px 0;
            line-height: 1.6;
            color: #34495e;
            font-size: 14px;
        }}
        .details-section p strong {{
            color: #2c3e50;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 13px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: right;
        }}
        .timestamp p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Prediction Output Report</h1>
        
        <h2>Overall Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Files Processed</td>
                    <td class="metric-value">{files_processed}</td>
                </tr>
                <tr>
                    <td>Total Person Objects</td>
                    <td class="metric-value">{person_count if person_count > 0 else 0}</td>
                </tr>
                <tr>
                    <td>Total Predictions</td>
                    <td class="metric-value">{total_predictions if total_predictions > 0 else 0}</td>
                </tr>
                <tr>
                    <td>Generated Date</td>
                    <td class="metric-value">{summary_data.get('generated_date', 'N/A')}</td>
                </tr>
            </tbody>
        </table>

        <h2>Per-File Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Filename</th>
                    <th>Total Tracks</th>
                    <th>Total Persons</th>
                </tr>
            </thead>
            <tbody>
"""

# Add per-file statistics
for filename in sorted(file_stats.keys()):
    stats = file_stats[filename]
    html_content += f"""                <tr>
                    <td>{filename}</td>
                    <td class="metric-value">{stats['total']}</td>
                    <td class="metric-value">{stats['persons']}</td>
                </tr>
"""

html_content += """            </tbody>
        </table>

        <h2>Prediction Results Details</h2>
        <div class="details-section">
"""

# Parse predictions.md and add details
if predictions_md_content:
    lines = predictions_md_content.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith('#'):
                # Skip headers as we're creating our own
                continue
            elif ':' in line:
                key, value = line.split(':', 1)
                html_content += f"<p><strong>{key.strip()}:</strong> {value.strip()}</p>\n"

html_content += """        </div>
        
        <div class="timestamp">
            <p>Report generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p>Source directory: """ + out_dir + """</p>
        </div>
    </div>
</body>
</html>
"""

# Save HTML report
html_output_path = os.path.join(out_dir, "prediction_report.html")
with open(html_output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

# Generate Markdown report as well
markdown_content = f"""# Prediction Output Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Source Directory:** {out_dir}

## Summary Metrics

| Metric | Value |
|--------|-------|
| Files Processed | {summary_data.get('files_processed', 'N/A')} |
| Total Person Objects | {person_count if person_count > 0 else 0} |
| Total Predictions | {total_predictions if total_predictions > 0 else 0} |
| Generated Date | {summary_data.get('generated_date', 'N/A')} |

## Prediction Results Details

"""

# Add predictions.md content to markdown report
if predictions_md_content:
    lines = predictions_md_content.split('\n')
    for line in lines:
        if line.strip() and not line.startswith('#'):
            markdown_content += line + "\n"

markdown_content += f"\n---\n*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

# Save Markdown report with new name
markdown_output_path = os.path.join(out_dir, "prediction-summary.md")
with open(markdown_output_path, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"\n✅ HTML report generated successfully: {html_output_path}")
print(f"✅ Markdown report generated successfully: {markdown_output_path}")
print(f"   Summary - Files: {summary_data.get('files_processed', 'N/A')}, "
      f"Persons: {person_count if person_count > 0 else 0}, "
      f"Total Predictions: {total_predictions if total_predictions > 0 else 0}")
    
