# Define the input and output file paths
input_file = "requirements.txt"
output_file = "cleaned_requirements.txt"

# Open the input file, process each line, and write to the output file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Strip whitespace and split on "==" to get the package name
        package = line.strip().split("==")[0]
        # Write the package name to the output file
        if package:  # Ensure it's not an empty line
            outfile.write(package + "\n")

print(f"Cleaned requirements saved to {output_file}")
