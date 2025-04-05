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
Install the requirements using:
pip install -r requirements.txt



