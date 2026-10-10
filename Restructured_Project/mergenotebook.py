import os
import nbformat


def merge_notebooks_to_py(input_folder, output_file="mergednotebook.py"):
    output_file = os.path.abspath(output_file)

    notebooks = []

    # Find all notebooks recursively
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".ipynb"):
                filepath = os.path.abspath(os.path.join(root, filename))

                # Don't include the output file if it is inside the folder
                if filepath != output_file:
                    notebooks.append(filepath)

    notebooks.sort()

    with open(output_file, "w", encoding="utf-8") as out:

        for filepath in notebooks:
            relative_path = os.path.relpath(filepath, input_folder)

            # File separator
            out.write("\n")
            out.write("# " + "=" * 80 + "\n")
            out.write(f"# NOTEBOOK: {relative_path}\n")
            out.write("# " + "=" * 80 + "\n\n")

            # Read notebook
            with open(filepath, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # Process cells
            for cell in notebook.cells:

                if cell.cell_type == "markdown":
                    # Convert Markdown to Python comments
                    for line in cell.source.splitlines():
                        out.write(f"# {line}\n")

                    out.write("\n")

                elif cell.cell_type == "code":
                    out.write(cell.source)
                    out.write("\n\n")

    print(f"Done!")
    print(f"Merged {len(notebooks)} notebooks")
    print(f"Output: {output_file}")


# Example
merge_notebooks_to_py(
    input_folder="notebooks",
    output_file="mergedcodes/mergednotebook.py"
)