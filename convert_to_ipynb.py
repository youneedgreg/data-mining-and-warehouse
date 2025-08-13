import os
import json
import re
from pathlib import Path

def py_to_ipynb(py_file_path, output_dir=None):
    """
    Convert a Python file to Jupyter notebook format
    """
    py_path = Path(py_file_path)
    
    # Set output directory
    if output_dir:
        output_path = Path(output_dir) / (py_path.stem + '.ipynb')
    else:
        output_path = py_path.with_suffix('.ipynb')
    
    # Read the Python file
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into cells based on comments or function definitions
    cells = split_into_cells(content)
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write the notebook file
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Converted {py_path.name} -> {output_path.name}")
    return output_path

def split_into_cells(content):
    """
    Split Python content into logical cells, converting comments to markdown
    """
    cells = []
    lines = content.split('\n')
    current_cell = []
    current_cell_type = 'code'
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for comment blocks (multiple consecutive comment lines)
        if line.strip().startswith('#'):
            # Save any existing code cell
            if current_cell and any(l.strip() for l in current_cell) and current_cell_type == 'code':
                cells.append(create_code_cell('\n'.join(current_cell)))
                current_cell = []
            
            # Collect all consecutive comment lines
            comment_block = []
            while i < len(lines) and lines[i].strip().startswith('#'):
                # Remove the # and any leading space
                comment_line = lines[i].strip()
                if comment_line == '#':
                    comment_block.append('')
                else:
                    comment_block.append(comment_line[1:].lstrip())
                i += 1
            
            # Create markdown cell from comments
            if comment_block and any(l.strip() for l in comment_block):
                cells.append(create_markdown_cell('\n'.join(comment_block)))
            
            current_cell = []
            current_cell_type = 'code'
            continue
        
        # Handle code lines
        if should_start_new_cell(line, current_cell):
            # Save previous cell if it has content
            if current_cell and any(l.strip() for l in current_cell):
                cells.append(create_code_cell('\n'.join(current_cell)))
            current_cell = [line]
        else:
            current_cell.append(line)
        
        current_cell_type = 'code'
        i += 1
    
    # Add the last cell
    if current_cell and any(l.strip() for l in current_cell):
        cells.append(create_code_cell('\n'.join(current_cell)))
    
    return cells

def should_start_new_cell(line, current_cell):
    """
    Determine if a line should start a new cell
    """
    stripped = line.strip()
    
    # Start new cell for major code blocks
    if (stripped.startswith('def ') or 
        stripped.startswith('class ') or
        stripped.startswith('if __name__')):
        return len(current_cell) > 0
    
    # Start new cell after imports section
    if current_cell and is_import_section_end(current_cell, line):
        return True
        
    return False

def is_import_section_end(current_cell, line):
    """
    Check if we're at the end of an import section
    """
    current_has_imports = any(l.strip().startswith(('import ', 'from ')) 
                             for l in current_cell if l.strip())
    line_is_not_import = (line.strip() and 
                         not line.strip().startswith(('import ', 'from ', '#')))
    
    return current_has_imports and line_is_not_import

def create_code_cell(source):
    """
    Create a code cell structure
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n')
    }

def create_markdown_cell(source):
    """
    Create a markdown cell structure
    """
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n')
    }

def batch_convert_directory(directory_path, output_dir=None):
    """
    Convert all Python files in a directory to notebooks
    """
    directory = Path(directory_path)
    py_files = list(directory.rglob('*.py'))
    
    if not py_files:
        print("No Python files found in the directory")
        return
    
    print(f"Found {len(py_files)} Python files to convert:")
    
    converted_files = []
    for py_file in py_files:
        try:
            output_path = py_to_ipynb(py_file, output_dir)
            converted_files.append(output_path)
        except Exception as e:
            print(f"Error converting {py_file.name}: {e}")
    
    print(f"\nSuccessfully converted {len(converted_files)} files!")
    return converted_files

# Example usage for your specific files
if __name__ == "__main__":
    # Convert specific files from your project structure
    files_to_convert = [
        "Section1_DataWarehousing/Task2_ETL/etl_retail.py",
        "Section1_DataWarehousing/Task3_OLAP/olap_queries.py",
        "Section2_DataMining/Task1_Preprocessing/preprocessing_iris.py",
        "Section2_DataMining/Task2_Clustering/clustering_iris.py",
        "Section2_DataMining/Task3_Classification_ARM/mining_iris_bucket.py"
    ]
    
    print("Converting Python files to Jupyter notebooks...")
    print("=" * 50)
    
    # Convert individual files
    converted_count = 0
    for file in files_to_convert:
        if os.path.exists(file):
            try:
                py_to_ipynb(file)
                converted_count += 1
            except Exception as e:
                print(f"Error converting {file}: {e}")
        else:
            print(f"File not found: {file}")
    
    print("=" * 50)
    print(f"Conversion complete! {converted_count} files converted.")
    
    # Alternative: Convert entire directory structure
    # Uncomment the line below to convert ALL .py files recursively:
    # batch_convert_directory(".", "notebooks_output")