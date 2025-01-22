import sys
import os
import linecache

# Define application name and data directory
data_dir = os.path.expanduser("~/Documents/Intelligent-Pipeline-Generator/")

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)

# Define file paths
output_file_path = os.path.join(data_dir, "trace_calls1.txt")
visited_file_path = os.path.join(data_dir, "visited.txt")
full_content_file_path = os.path.join(data_dir, "trace_calls0.txt")

# Open the files in write mode
output_file = open(output_file_path, "w")
visited_file = open(visited_file_path, "w")
full_content_file = open(full_content_file_path, "w")

executed_lines = set()  # Set to store unique executed lines (filename, lineno)
executed_code_lines = []  # List to preserve the order of execution
visited_files = set()  # Set to store unique visited files

# Directories/keywords to exclude (e.g., environment folders, site-packages)
excluded_directories = ["venv", "env", ".venv", "site-packages", "dist-packages"]


def should_trace(filename, target_script_directory):
    """
    Returns True if 'filename' should be traced (i.e., is in the target folder
    and not in excluded directories), otherwise False.
    """
    # Ensure the file is within the target_script_directory
    if not filename.startswith(target_script_directory):
        return False

    # Check for excluded directories or keywords in the filename's path
    for exclude in excluded_directories:
        if exclude in filename:
            return False

    return True


def trace_functions(frame, event, arg):
    if event == "call":
        code = frame.f_code
        filename = code.co_filename

        # Trace only if file is within the project folder and not excluded
        if should_trace(filename, target_script_directory):
            rel_path = os.path.relpath(filename, target_script_directory)
            visited_files.add(rel_path)

    elif event == "line":
        code = frame.f_code
        filename = code.co_filename
        lineno = frame.f_lineno

        # Record executed lines only for files in project folder and not excluded
        if should_trace(filename, target_script_directory):
            if (filename, lineno) not in executed_lines:
                line = linecache.getline(filename, lineno)
                if line.strip() and not line.strip().startswith("#"):
                    executed_lines.add((filename, lineno))
                    executed_code_lines.append(line.rstrip())

    return trace_functions


if __name__ == "__main__":
    # Ensure command-line arguments are passed
    if len(sys.argv) < 2:
        print("Usage: python tracer.py <target_script> [args...]")
        sys.exit(1)

    # Path to the target script and its directory
    target_script_file = os.path.abspath(sys.argv[1])
    target_script_directory = os.path.dirname(target_script_file)

    # Set up tracing
    sys.settrace(trace_functions)

    # Prepare the arguments for the target script
    target_script_args = sys.argv[1:]
    sys.argv = target_script_args

    # Run the target script using exec
    with open(target_script_file, "rb") as file:
        exec(compile(file.read(), target_script_file, "exec"))

    sys.settrace(None)  # Disable tracing after execution

    # Write the collected code lines to the output file (trace_calls0.txt)
    for line in executed_code_lines:
        output_file.write(line + "\n")
    output_file.close()

    # Write visited files to visited.txt
    for file in visited_files:
        visited_file.write(file + "\n")
    visited_file.close()

    # Now write the full content of each visited file to trace_calls1.txt
    # (only files within the target project directory, excluding environment)
    for file in visited_files:
        abs_path = os.path.join(target_script_directory, file)
        if os.path.isfile(abs_path):
            with open(abs_path, "r") as f:
                content = f.read()
            full_content_file.write(f"{file}:\n\n{content}\n\n")

    full_content_file.close()
