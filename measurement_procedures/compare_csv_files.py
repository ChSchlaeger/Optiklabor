import csv


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

    differences = []

    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        reader1 = csv.reader(f1, delimiter=';')
        reader2 = csv.reader(f2, delimiter=';')

        row_num = 1
        for row1, row2 in zip(reader1, reader2):
            if len(row1) != len(row2):
                differences.append((row_num, None, None, f"Row length differs: {len(row1)} vs {len(row2)}"))
            else:
                for col_num, (val1, val2) in enumerate(zip(row1, row2), start=1):
                    parsed_val1 = parse_value(val1)
                    parsed_val2 = parse_value(val2)

                    if parsed_val1 != parsed_val2:
                        differences.append((row_num, col_num, parsed_val1, parsed_val2))
            row_num += 1

    if differences:
        print(f"Found {len(differences)} differences:")
        for diff in differences:
            if diff[1] is None:  # Row length difference
                print(f"Row {diff[0]}: {diff[3]}")
            else:
                print(f"Row {diff[0]}, Column {diff[1]}: {diff[2]} != {diff[3]}")
    else:
        print("The files are identical (considering the allowed float/int discrepancy).")

    return differences


# Example usage
file1 = "BRDF_measurement_total_2416101param1_gamma-plus-360.csv"
file2 = "JAN_BRDF_Nov24_measurement_total_24_16_10_1_param1_gamma-plus-360.csv"
compare_csv_files(file1, file2)
