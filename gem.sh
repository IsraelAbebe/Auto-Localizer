#!/bin/bash

# --- Script Configuration ---
# Name of your Python localization script
PYTHON_SCRIPT_NAME="gem.py" 

# --- Help Function ---
usage() {
  echo "Usage: $0 [-g | -e] [-n num_samples] [-v] [-o output_dataset_name] [-h]"
  echo ""
  echo "Or, for more advanced options, run: python $PYTHON_SCRIPT_NAME --help"
  echo ""
  echo "  -g                  : Enable 'generate' mode (maps to --generate in Python script)"
  echo "  -e                  : Enable 'experiment' mode (maps to --experiment in Python script)"
  echo "                      (NOTE: -g and -e are mutually exclusive; -e takes precedence if both are used)"
  echo "  -n <num_samples>    : Number of samples to process (maps to --num_samples)"
  echo "  -v                  : Enable verbose logging (maps to --verbose)"
  echo "  -o <dataset_name>   : Specify the output dataset name for Hugging Face Hub (maps to --output_dataset_name)"
  echo "  -h                  : Display this help message"
  echo ""
  echo "Any additional arguments not listed above will be passed directly to the Python script."
  exit 1
}

# --- Default Values for Flags ---
GENERATE_MODE=false
EXPERIMENT_MODE=false
NUM_SAMPLES_SET=false # Track if -n was explicitly set
NUM_SAMPLES=""        # Value for --num_samples
VERBOSE_MODE=false
OUTPUT_DATASET_NAME=""

# --- Parse Command-Line Options ---
# Initialize an array to hold arguments specifically for the Python script
PYTHON_ARGS=()

# Using getopts for robust option parsing
# The colon after an option means it requires an argument (e.g., n:, o:)
while getopts ":gen:vo:h" opt; do
  case ${opt} in
    g )
      GENERATE_MODE=true
      ;;
    e )
      EXPERIMENT_MODE=true
      ;;
    n )
      # Validate that NUM_SAMPLES is a positive integer
      if ! [[ "$OPTARG" =~ ^[0-9]+$ ]] || [ "$OPTARG" -lt 0 ]; then
        echo "Error: Option -n requires a non-negative integer for num_samples." >&2
        usage
      fi
      NUM_SAMPLES_SET=true
      NUM_SAMPLES="$OPTARG"
      ;;
    v )
      VERBOSE_MODE=true
      ;;
    o )
      OUTPUT_DATASET_NAME="$OPTARG"
      ;;
    h )
      usage # Call the help function
      ;;
    \? )
      echo "Error: Invalid option: -$OPTARG" >&2
      usage
      ;;
    : )
      echo "Error: Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Shift off the options that getopts has already processed
shift $((OPTIND -1))

# --- Construct Python Script Arguments ---

# Experiment mode takes precedence over generate mode if both are selected
if [ "$EXPERIMENT_MODE" = true ]; then
  PYTHON_ARGS+=("--experiment")
elif [ "$GENERATE_MODE" = true ]; then
  PYTHON_ARGS+=("--generate")
fi

# Add num_samples if it was provided
if [ "$NUM_SAMPLES_SET" = true ]; then
  PYTHON_ARGS+=("--num_samples" "$NUM_SAMPLES")
fi

# Add verbose flag if enabled
if [ "$VERBOSE_MODE" = true ]; then
  PYTHON_ARGS+=("--verbose")
fi

# Add output_dataset_name if provided
if [ -n "$OUTPUT_DATASET_NAME" ]; then
  PYTHON_ARGS+=("--output_dataset_name" "$OUTPUT_DATASET_NAME")
fi

# Add any remaining arguments from the command line directly to the Python script
PYTHON_ARGS+=("$@")

# --- Pre-execution Checks ---

# Verify the Python script exists
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
  echo "Error: Python script '$PYTHON_SCRIPT_NAME' not found in the current directory." >&2
  exit 1
fi

# --- Execute the Python Script ---

echo "Executing: python $PYTHON_SCRIPT_NAME ${PYTHON_ARGS[@]}"
python "$PYTHON_SCRIPT_NAME" "${PYTHON_ARGS[@]}"

# --- Completion Message ---
echo "Script execution finished."