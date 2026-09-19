import re
import csv

INPUT_FILE = "gsi_data.txt"
OUTPUT_FILE = "gsi_ner_landslides.csv"

NER_STATES = [
    "Arunachal Pradesh",
    "Assam",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura"
]

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# PDF extraction may split one record across multiple lines.
text = re.sub(r"\s+", " ", text)

records = []

# Find latitude/longitude pairs that fall within the North-East India region.
coord_pattern = re.compile(
    r"\b(2[2-9]\.\d{3,})\s+(8[8-9]\.\d{3,}|9[0-8]\.\d{3,})\b"
)

for match in coord_pattern.finditer(text):
    latitude = float(match.group(1))
    longitude = float(match.group(2))

    # Keep surrounding text so we can identify the state and record details.
    start = max(0, match.start() - 300)
    end = min(len(text), match.end() + 150)
    context = text[start:end]

    state = None

    for s in NER_STATES:
        if s.lower() in context.lower():
            state = s
            break

    if state:
        records.append({
            "state": state,
            "latitude": latitude,
            "longitude": longitude,
            "source": "GSI Landslide Inventory",
            "raw_record": context
        })

# Remove duplicate coordinates.
unique_records = []
seen = set()

for record in records:
    key = (
        record["state"],
        record["latitude"],
        record["longitude"]
    )

    if key not in seen:
        seen.add(key)
        unique_records.append(record)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "state",
            "latitude",
            "longitude",
            "source",
            "raw_record"
        ]
    )

    writer.writeheader()
    writer.writerows(unique_records)

print("--------------------------------")
print("StrataNex GSI Data Extraction")
print("--------------------------------")
print("Records found:", len(unique_records))
print("Output:", OUTPUT_FILE)

for state in NER_STATES:
    count = sum(1 for r in unique_records if r["state"] == state)
    print(state, ":", count)