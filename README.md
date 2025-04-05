# Einops Implementation from Scratch
This project implements the core functionality of the einops library from scratch, specifically focusing on the rearrange operation. 


The implementation consists of these main components:

1. Pattern Parsing- Parse the pattern string to identify input and output patterns, and extract axis names.
2. Dimension Inference-Determine the sizes of axes based on the input tensor shape and any provided axes lengths.
3. Axis Manipulation-Handle operations like reshaping, transposition, splitting, merging, and repeating of axes.
4. Error Checking-Provide informative error messages for various failure cases.

## Design Decisions
- The implementation uses a staged approach:
  1. Parse the pattern
  2. Identify axis dimensions
  3. Handle composites in the input (reshape if needed)
  4. Prepare for the final output shape
  5. Perform transposition
  6. Final reshaping
  
## How to Run
Clone this repository to get started:
git clone <repository-url>
cd <repository-folder>

## After cloning 
1. Create a separate Evironment for this project
cmd: conda create -p env_name(venv used by me) python==version (in this i have used 3.10)
2. Activate the env using : conda activate env_name/
3. Install the requirements using:
pip install -r requirements.txt

## Running the code
### Run these commands in the cmd terminal 
1. python Einops_module.py
2. python UnitTest.py



