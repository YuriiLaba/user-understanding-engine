import json
import csv

def convert_propositions_to_csv(input_path: str, output_path: str):
    """
    Converts a JSONL file with propositions into a CSV file.
    
    Each JSONL entry must contain a 'propositions' list with
    'proposition', 'confidence', and 'decay' fields.
    """
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["proposition", "confidence", "decay"])  # header

        for line in infile:
            try:
                entry = json.loads(line.strip())
                for prop in entry.get("propositions", []):
                    
                        writer.writerow([
                            prop.get("proposition", ""),
                            prop.get("confidence", ""),
                            prop.get("decay", "")
                        ])
            except Exception as e:
                print(f"Error writing proposition {prop}: {e}")
                continue
    print(f"CSV file created at: {output_path}")