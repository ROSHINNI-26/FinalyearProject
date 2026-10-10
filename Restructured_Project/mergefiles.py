import os


def merge_python_files(input_folder, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:

        for filename in sorted(os.listdir(input_folder)):
            if filename.endswith(".py") and filename != os.path.basename(output_file):

                filepath = os.path.join(input_folder, filename)

                outfile.write(f"\n# {'=' * 70}\n")
                outfile.write(f"# FILE: {filename}\n")
                outfile.write(f"# {'=' * 70}\n\n")

                with open(filepath, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())

                outfile.write("\n\n")


# Example
merge_python_files(
    input_folder="src",
    output_file="mergedcodes/mergedsrc.py"
)
