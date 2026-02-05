#!/bin/bash
# Cross-platform Virtual Environment Activation Script
# Works on Linux, macOS, and Windows (Git Bash/WSL)
#
# Usage options:
#   1. source ./activate_venv.sh  (recommended - works directly)
#   2. . ./activate_venv.sh       (shorthand for source)
#   3. eval "$(./activate_venv.sh)" (works when executed directly)
#
# Note: Running ./activate_venv.sh directly will show instructions
#       because child processes cannot modify parent shell environment

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if script is being sourced or executed directly
# When sourced: ${BASH_SOURCE[0]} != $0
# When executed: ${BASH_SOURCE[0]} == $0
# Note: When executed directly, we can't modify the parent shell's environment
# So we'll find the venv and source it directly, which will work if the user
# runs: eval "$(./activate_venv.sh)" or we output instructions
IS_SOURCED=false
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    IS_SOURCED=true
fi

# Function to find venv directory
find_venv() {
    local current_dir="$PWD"
    
    # Check current directory
    if [ -d "$current_dir/venv" ]; then
        echo "$current_dir/venv"
        return 0
    fi
    
    # Check parent directory
    if [ -d "$current_dir/../venv" ]; then
        echo "$current_dir/../venv"
        return 0
    fi
    
    # Check for .venv
    if [ -d "$current_dir/.venv" ]; then
        echo "$current_dir/.venv"
        return 0
    fi
    
    if [ -d "$current_dir/../.venv" ]; then
        echo "$current_dir/../.venv"
        return 0
    fi
    
    return 1
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)
            echo "linux"
            ;;
        Darwin*)
            echo "macos"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Main activation logic
OS=$(detect_os)
VENV_PATH=$(find_venv)

if [ -z "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}" >&2
    echo -e "${YELLOW}Please ensure you have created a virtual environment named 'venv' or '.venv'${NC}" >&2
    echo -e "${YELLOW}You can create one with: python3 -m venv venv${NC}" >&2
    exit 1
fi

# Normalize path (resolve relative paths)
VENV_PATH=$(cd "$VENV_PATH" && pwd)

# Determine activation script path
if [ "$OS" = "windows" ]; then
    # Windows (Git Bash/WSL)
    if [ -f "$VENV_PATH/Scripts/activate" ]; then
        ACTIVATE_SCRIPT="$VENV_PATH/Scripts/activate"
    elif [ -f "$VENV_PATH/bin/activate" ]; then
        ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
    else
        echo -e "${RED}Error: Activation script not found in $VENV_PATH${NC}" >&2
        exit 1
    fi
else
    # Linux/macOS
    if [ -f "$VENV_PATH/bin/activate" ]; then
        ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
    else
        echo -e "${RED}Error: Activation script not found in $VENV_PATH${NC}" >&2
        exit 1
    fi
fi

# Activate based on whether script is sourced or executed
if [ "$IS_SOURCED" = true ]; then
    # Script is being sourced - activate directly
    if [ "$OS" = "windows" ]; then
        echo -e "${GREEN}Activating virtual environment (Windows)...${NC}"
    else
        echo -e "${GREEN}Activating virtual environment ($OS)...${NC}"
    fi
    source "$ACTIVATE_SCRIPT"
    
    # Verify activation
    if [ -n "$VIRTUAL_ENV" ]; then
        echo -e "${GREEN}Virtual environment activated successfully!${NC}"
        echo -e "${GREEN}Python: $(which python)${NC}"
        echo -e "${GREEN}Virtual Env: $VIRTUAL_ENV${NC}"
    else
        echo -e "${YELLOW}Warning: Virtual environment may not have activated correctly.${NC}"
        echo -e "${YELLOW}Try running: source $VENV_PATH/bin/activate${NC}"
    fi
else
    # Script is being executed directly - output the source command
    # User can run: eval "$(./activate_venv.sh)" to activate
    echo -e "${GREEN}Virtual environment found: $VENV_PATH${NC}" >&2
    echo -e "${YELLOW}To activate, run: eval \"\$(./activate_venv.sh)\"${NC}" >&2
    echo -e "${YELLOW}Or use: source ./activate_venv.sh${NC}" >&2
    echo ""
    echo "source '$ACTIVATE_SCRIPT'"
fi
