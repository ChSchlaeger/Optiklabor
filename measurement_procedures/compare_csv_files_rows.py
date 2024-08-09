import csv
from collections import defaultdict


def compare_csv_files(file1, file2):
    def parse_value(value):
        """Convert a string value to an appropriate data type."""
        if value == "":
            return None
        value = value.replace(",", ".")
        try:
            # Attempt to convert to float
            float_val = float(value)
            # Return as int if the value is a whole number
            return int(float_val) if float_val.is_integer() else float_val
        except ValueError:
            # Return as string if conversion fails
            return value

    def normalize_row(row):
        """Normalize a row to handle float/int discrepancies."""
        return tuple(parse_value(val) for val in row)

    def read_csv(file):
        """Read CSV file and return a dictionary of normalized rows."""
        rows = defaultdict(int)
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                normalized = normalize_row(row)
                rows[normalized] += 1
        return rows

    rows1 = read_csv(file1)
    rows2 = read_csv(file2)

    differences = []

    # Find differences by comparing row counts
    for row, count in rows1.items():
        if row not in rows2:
            differences.append(f"Row {row} is missing in {file2}")
        elif count != rows2[row]:
            differences.append(f"Row {row} count differs: {file1} has {count}, {file2} has {rows2[row]}")

    for row, count in rows2.items():
        if row not in rows1:
            differences.append(f"Row {row} is missing in {file1}")

    if differences:
        print(f"Found {len(differences)} differences:")
        for diff in differences:
            print(diff)
    else:
        print("The files are identical (considering allowed float/int discrepancy and row reordering).")

    return differences

# Example usage
file1 = "BRDF_measurement_total_2416101param1_gamma-plus-360_old.csv"
file2 = "JAN_BRDF_Nov24_measurement_total_24_16_10_1_param1_gamma-plus-360.csv"
compare_csv_files(file1, file2)
