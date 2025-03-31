import re
import csv

# Define the regex patterns for each line we want to extract.
episode_header_re = re.compile(
    r"Episode\s+(\d+)/\d+\s+\|\s+Step\s+(\d+)\s+\|\s+Score:\s+([\d,]+)\s+\|\s+Max Tile:\s+([\d.]+)"
)
completed_re = re.compile(
    r"Episode\s+\d+/\d+\s+completed in\s+([\d.]+)s"
)
final_re = re.compile(
    r"Final Score:\s+([\d,]+)\s+\|\s+Steps:\s+(\d+)\s+\|\s+Max Tile:\s+([\d.]+)"
)
running_avg_re = re.compile(
    r"Running Average Score:\s+([\d,]+)"
)

# Input and output file names
input_file = "expectimax.log"
output_file = "expectimax_data.csv"

# List to hold rows (each row is a dict for CSV)
rows = []

with open(input_file, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    # Look for an episode header line
    header_match = episode_header_re.match(line)
    if header_match:
        episode_num = header_match.group(1)
        header_steps = header_match.group(2)
        header_score = header_match.group(3).replace(",", "")
        header_max_tile = header_match.group(4)

        # The next lines should follow: completed, final, running avg.
        # Use try/except if log formatting might vary.
        completed_line = lines[i+1].strip() if i+1 < len(lines) else ""
        final_line = lines[i+2].strip() if i+2 < len(lines) else ""
        running_avg_line = lines[i+3].strip() if i+3 < len(lines) else ""

        completed_match = completed_re.match(completed_line)
        final_match = final_re.match(final_line)
        running_avg_match = running_avg_re.match(running_avg_line)

        completed_time = completed_match.group(1) if completed_match else ""
        final_score = final_match.group(1).replace(",", "") if final_match else ""
        final_steps = final_match.group(2) if final_match else ""
        final_max_tile = final_match.group(3) if final_match else ""
        running_avg = running_avg_match.group(1).replace(",", "") if running_avg_match else ""

        row = {
            "Episode": episode_num,
            "Header Steps": header_steps,
            "Header Score": header_score,
            "Header Max Tile": header_max_tile,
            "Completed Time (s)": completed_time,
            "Final Score": final_score,
            "Final Steps": final_steps,
            "Final Max Tile": final_max_tile,
            "Running Average Score": running_avg,
        }
        rows.append(row)
        # Skip the block lines (assuming 4 lines + separator line, so jump 5 lines)
        # Adjust jump if needed (here we skip the 4 lines that we processed and let the loop continue)
        i += 5
        continue
    else:
        i += 1

# Write rows to CSV
with open(output_file, "w", newline="") as csvfile:
    fieldnames = [
        "Episode",
        "Header Steps",
        "Header Score",
        "Header Max Tile",
        "Completed Time (s)",
        "Final Score",
        "Final Steps",
        "Final Max Tile",
        "Running Average Score",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"CSV file '{output_file}' has been created with {len(rows)} episodes.")
