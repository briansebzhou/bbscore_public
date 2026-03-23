#!/bin/bash
#
# BBScore Installation Script
#
# This script sets up the BBScore environment for students.
# It handles conda installation, environment creation, and dependency installation.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh              # Interactive mode (recommended)
#   ./install.sh --quick      # Skip interactive setup, use defaults
#   ./install.sh --help       # Show all options
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

ENV_NAME="bbscore"
PYTHON_VERSION="3.11"
CPU_ONLY=false
SKIP_CONDA=false
DATA_DIR=""
NON_INTERACTIVE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIN_DISK_SPACE_GB=10
ENV_PREFIX=""
PIP_CACHE_DIR=""
PIP_TMP_DIR=""
ACTIVATE_SUFFIX=""
CLEAN_CACHE=false

# ============================================================================
# Color Support Detection
# ============================================================================

if [ -t 1 ] && [ -n "$TERM" ] && [ "$TERM" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    DIM=''
    NC=''
fi

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Get available disk space in GB
get_available_disk_space() {
    local path="$1"
    if [ "$OS" = "macos" ]; then
        df -g "$path" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0"
    else
        df -BG "$path" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4}' || echo "0"
    fi
}

# Download with retry logic
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if command_exists curl; then
            if curl -fsSL --retry 3 --connect-timeout 30 -o "$output" "$url"; then
                return 0
            fi
        elif command_exists wget; then
            if wget -q --timeout=30 --tries=3 -O "$output" "$url"; then
                return 0
            fi
        else
            print_error "Neither curl nor wget found. Please install one of them."
            exit 1
        fi

        retry=$((retry + 1))
        print_warning "Download failed, attempt $retry of $max_retries..."
        sleep 2
    done

    return 1
}

# Cleanup function for error handling
cleanup_on_error() {
    echo ""
    print_error "Installation failed!"
    print_info "You can try running the script again or check the error above."
    exit 1
}

trap cleanup_on_error ERR

# ============================================================================
# Interactive Menu Functions
# ============================================================================

# Display a selection menu and return the choice
# Usage: selected=$(show_menu "Title" "option1" "option2" "option3")
show_menu() {
    local title="$1"
    shift
    local options=("$@")
    local selected=0
    local key=""

    # Hide cursor
    tput civis 2>/dev/null || true

    # Restore cursor on exit
    trap 'tput cnorm 2>/dev/null || true' RETURN

    while true; do
        # Clear previous menu (output to stderr)
        for ((i=0; i<${#options[@]}+2; i++)); do
            tput cuu1 2>/dev/null || true
            tput el 2>/dev/null || true
        done 2>/dev/null || true

        # Print title (to stderr so it doesn't get captured)
        echo -e "${BOLD}${title}${NC}" >&2
        echo -e "${DIM}(Use arrow keys or j/k to navigate, Enter to select)${NC}" >&2

        # Print options
        for i in "${!options[@]}"; do
            if [ $i -eq $selected ]; then
                echo -e "  ${GREEN}▸ ${options[$i]}${NC}" >&2
            else
                echo -e "    ${options[$i]}" >&2
            fi
        done

        # Read single keypress
        read -rsn1 key 2>/dev/null || read -rs -k1 key 2>/dev/null || true

        # Handle arrow keys (escape sequences)
        if [[ $key == $'\x1b' ]]; then
            # Read rest of escape sequence (works in both bash and zsh)
            read -rsn2 -t 1 key 2>/dev/null || read -rs -k2 -t 1 key 2>/dev/null || true
            case "$key" in
                '[A') # Up arrow
                    ((selected > 0)) && ((selected--))
                    ;;
                '[B') # Down arrow
                    ((selected < ${#options[@]}-1)) && ((selected++))
                    ;;
            esac
        else
            case "$key" in
                'k'|'K') # Vim up
                    ((selected > 0)) && ((selected--))
                    ;;
                'j'|'J') # Vim down
                    ((selected < ${#options[@]}-1)) && ((selected++))
                    ;;
                '') # Enter
                    tput cnorm 2>/dev/null || true
                    echo "$selected"
                    return
                    ;;
                [0-9]) # Number selection
                    if [ "$key" -lt "${#options[@]}" ] 2>/dev/null; then
                        selected=$key
                        tput cnorm 2>/dev/null || true
                        echo "$selected"
                        return
                    fi
                    ;;
            esac
        fi
    done
}

# Simple fallback menu for non-interactive terminals
show_simple_menu() {
    local title="$1"
    shift
    local options=("$@")

    echo -e "${BOLD}${title}${NC}" >&2
    for i in "${!options[@]}"; do
        echo "  $((i+1)). ${options[$i]}" >&2
    done

    while true; do
        echo -n "Enter choice [1-${#options[@]}]: " >&2
        read choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
            echo "$((choice-1))"
            return
        fi
        echo "Invalid choice. Please enter a number between 1 and ${#options[@]}." >&2
    done
}

# Wrapper that chooses the right menu type
select_option() {
    local title="$1"
    shift
    local options=("$@")

    if [ "$NON_INTERACTIVE" = true ]; then
        echo "0"  # Return first option in non-interactive mode
        return
    fi

    # Check if we can use the fancy menu (need tput and terminal)
    if [ -t 0 ] && [ -t 1 ] && command_exists tput; then
        # Print initial lines for the menu to overwrite (to stderr)
        echo "" >&2
        echo "" >&2
        for ((i=0; i<${#options[@]}; i++)); do
            echo "" >&2
        done
        show_menu "$title" "${options[@]}"
    else
        show_simple_menu "$title" "${options[@]}"
    fi
}

# Prompt for text input with tab completion for paths
# Usage: result=$(prompt_path "Enter directory" "/default/path")
prompt_path() {
    local prompt="$1"
    local default="$2"
    local result

    if [ "$NON_INTERACTIVE" = true ]; then
        echo "$default"
        return
    fi

    # Output prompts to stderr so they don't get captured by command substitution
    echo -e "${BOLD}${prompt}${NC}" >&2
    echo -e "${DIM}(Tab completion enabled, press Enter for default)${NC}" >&2
    echo -e "${DIM}Default: ${default}${NC}" >&2

    # Use read -e for readline support (tab completion)
    # Note: -i option not supported in zsh, so we show default separately
    read -e -p "> " result

    # Use default if empty
    result="${result:-$default}"

    # Expand tilde
    result="${result/#\~/$HOME}"

    echo "$result"
}

# Prompt for text input (no tab completion)
prompt_text() {
    local prompt="$1"
    local default="$2"
    local result

    if [ "$NON_INTERACTIVE" = true ]; then
        echo "$default"
        return
    fi

    if [ -n "$default" ]; then
        # Output to stderr to avoid capture by command substitution
        echo -n "$prompt [$default]: " >&2
        read result
        result="${result:-$default}"
    else
        echo -n "$prompt: " >&2
        read result
    fi

    echo "$result"
}

# Yes/No prompt
prompt_yes_no() {
    local prompt="$1"
    local default="$2"

    if [ "$NON_INTERACTIVE" = true ]; then
        [ "$default" = "y" ] && return 0 || return 1
    fi

    local yn_hint="y/N"
    [ "$default" = "y" ] && yn_hint="Y/n"

    read -p "$prompt ($yn_hint): " -n 1 -r
    echo

    if [ -z "$REPLY" ]; then
        [ "$default" = "y" ] && return 0 || return 1
    fi

    [[ $REPLY =~ ^[Yy]$ ]] && return 0 || return 1
}

# ============================================================================
# Conda Health Check
# ============================================================================

check_conda_health() {
    if command_exists conda; then
        if ! conda --version &> /dev/null 2>&1; then
            print_warning "Conda appears to be broken (likely jaraco.functools issue)"
            print_info "Attempting to fix..."

            local conda_bin_dir
            conda_bin_dir=$(dirname "$(which conda)")
            local conda_python="$conda_bin_dir/../bin/python"

            if [ ! -f "$conda_python" ]; then
                conda_python="$conda_bin_dir/python"
            fi

            if [ -f "$conda_python" ]; then
                "$conda_python" -m pip install --upgrade jaraco.functools 2>/dev/null || \
                "$conda_python" -m pip install 'setuptools<71' 2>/dev/null || true
            fi

            if ! conda --version &> /dev/null 2>&1; then
                print_error "Could not fix conda automatically."
                print_info "Please try running manually:"
                echo "  \$(conda info --base)/bin/python -m pip install --upgrade jaraco.functools"
                exit 1
            fi
            print_success "Conda fixed successfully!"
        fi
    fi
}

# ============================================================================
# Help
# ============================================================================

show_help() {
    cat << EOF
${BOLD}BBScore Installation Script${NC}

${BOLD}Usage:${NC}
  ./install.sh                  Interactive setup (recommended for students)
  ./install.sh --quick          Use all defaults, minimal prompts
  ./install.sh [OPTIONS]        Custom installation

${BOLD}Options:${NC}
  --quick               Skip interactive setup, use smart defaults
  --non-interactive     Same as --quick (for CI/scripts)
  --cpu-only            Install CPU-only PyTorch (no CUDA)
  --data-dir DIR        Set data directory (default: ~/bbscore_data)
  --env-name NAME       Set environment name (default: bbscore)
  --env-prefix DIR      Create conda env at prefix path (default: named env)
  --pip-dirs DIR        Redirect pip cache/tmp under DIR (e.g. project dir)
  --activate-suffix S   Activation script suffix (e.g. peggyo -> activate_bbscore_peggyo.sh)
  --clean               Remove .pip_cache at end (keeps .pip_tmp cleaned by default)
  --no-conda            Use pip/venv instead of conda
  --help, -h            Show this help message

${BOLD}Examples:${NC}
  ./install.sh                              # Interactive setup wizard
  ./install.sh --quick                      # Quick install with defaults
  ./install.sh --quick --cpu-only           # Quick CPU-only install
  ./install.sh --data-dir /data/bbscore     # Custom data location

${BOLD}Interactive Features:${NC}
  - Arrow key menu navigation (or j/k for vim users)
  - Tab completion for directory paths
  - Smart defaults based on your system

EOF
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-conda)
            SKIP_CONDA=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --data-dir)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                print_error "--data-dir requires a directory path"
                exit 1
            fi
            DATA_DIR="$2"
            shift 2
            ;;
        --env-name)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                print_error "--env-name requires a name"
                exit 1
            fi
            ENV_NAME="$2"
            shift 2
            ;;
        --env-prefix)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                print_error "--env-prefix requires a directory path"
                exit 1
            fi
            ENV_PREFIX="$2"
            shift 2
            ;;
        --pip-dirs)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                print_error "--pip-dirs requires a directory path"
                exit 1
            fi
            PIP_CACHE_DIR="$2/.pip_cache"
            PIP_TMP_DIR="$2/.pip_tmp"
            shift 2
            ;;
        --activate-suffix)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                print_error "--activate-suffix requires a value"
                exit 1
            fi
            ACTIVATE_SUFFIX="$2"
            shift 2
            ;;
        --clean)
            CLEAN_CACHE=true
            shift
            ;;
        --quick|--non-interactive)
            NON_INTERACTIVE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# ============================================================================
# System Detection (needed before interactive setup)
# ============================================================================

OS="unknown"
ARCH=$(uname -m)

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

# Detect GPU
HAS_NVIDIA_GPU=false
if command_exists nvidia-smi; then
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
        HAS_NVIDIA_GPU=true
    fi
fi

HAS_APPLE_SILICON=false
if [ "$OS" = "macos" ] && [ "$ARCH" = "arm64" ]; then
    HAS_APPLE_SILICON=true
fi

# ============================================================================
# Interactive Setup Wizard
# ============================================================================

run_interactive_setup() {
    clear 2>/dev/null || true

    echo ""
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║                                                            ║${NC}"
    echo -e "${BOLD}${BLUE}║              🧠  BBScore Setup Wizard  🧠                  ║${NC}"
    echo -e "${BOLD}${BLUE}║                                                            ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    echo -e "${DIM}This wizard will help you configure BBScore for your system.${NC}"
    echo -e "${DIM}Detected: $OS ($ARCH)${NC}"
    if [ "$HAS_NVIDIA_GPU" = true ]; then
        echo -e "${DIM}GPU: NVIDIA GPU detected${NC}"
    elif [ "$HAS_APPLE_SILICON" = true ]; then
        echo -e "${DIM}GPU: Apple Silicon (MPS) detected${NC}"
    else
        echo -e "${DIM}GPU: No GPU detected (CPU mode)${NC}"
    fi
    echo ""

    # Step 1: PyTorch Configuration
    print_header "Step 1/4: PyTorch Configuration"

    if [ "$HAS_NVIDIA_GPU" = true ]; then
        echo -e "NVIDIA GPU detected! You can use GPU acceleration."
        echo ""
        choice=$(select_option "Select PyTorch version:" \
            "GPU (CUDA) - Recommended for your system" \
            "CPU only - Smaller download, no GPU acceleration")

        [ "$choice" -eq 1 ] && CPU_ONLY=true
    elif [ "$HAS_APPLE_SILICON" = true ]; then
        echo -e "Apple Silicon detected! MPS acceleration will be enabled automatically."
        echo ""
        CPU_ONLY=false
    else
        echo -e "No GPU detected. Installing CPU version of PyTorch."
        echo ""
        CPU_ONLY=true
    fi

    # Step 2: Environment Name
    print_header "Step 2/4: Environment Name"

    echo -e "The conda environment name for BBScore."
    echo -e "${DIM}Use a unique name if you have other Python projects.${NC}"
    echo ""
    ENV_NAME=$(prompt_text "Environment name" "bbscore")

    # Step 3: Data Directory
    print_header "Step 3/4: Data Directory"

    echo -e "BBScore needs a directory to store datasets and model weights."
    echo -e "${YELLOW}This directory should have at least 50GB of free space.${NC}"
    echo ""

    DEFAULT_DATA_DIR="$HOME/bbscore_data"

    # Show available space
    AVAILABLE_SPACE=$(get_available_disk_space "$HOME")
    echo -e "${DIM}Available space in home directory: ${AVAILABLE_SPACE}GB${NC}"
    echo ""

    DATA_DIR=$(prompt_path "Data directory" "$DEFAULT_DATA_DIR")

    # Step 4: Summary and Confirmation
    print_header "Step 4/4: Confirm Configuration"

    echo -e "${BOLD}Installation Summary:${NC}"
    echo ""
    echo -e "  Environment name:  ${GREEN}$ENV_NAME${NC}"
    echo -e "  Python version:    ${GREEN}$PYTHON_VERSION${NC}"
    echo -e "  PyTorch:           ${GREEN}$([ "$CPU_ONLY" = true ] && echo "CPU only" || echo "GPU enabled")${NC}"
    echo -e "  Data directory:    ${GREEN}$DATA_DIR${NC}"
    echo ""

    if ! prompt_yes_no "Proceed with installation?" "y"; then
        echo ""
        print_info "Installation cancelled."
        exit 0
    fi
}

# ============================================================================
# Main Installation
# ============================================================================

# Run interactive setup if not in non-interactive mode and no custom flags
if [ "$NON_INTERACTIVE" = false ] && [ -z "$DATA_DIR" ]; then
    run_interactive_setup
fi

print_header "BBScore Installation"

echo "Configuration:"
echo "  Environment: $ENV_NAME"
echo "  Python: $PYTHON_VERSION"
echo "  PyTorch: $([ "$CPU_ONLY" = true ] && echo "CPU only" || echo "GPU enabled")"
echo "  Data dir: ${DATA_DIR:-$HOME/bbscore_data}"
echo ""

# ============================================================================
# Step 0: System Checks
# ============================================================================

print_header "Checking System Requirements"

print_info "OS: $OS ($ARCH)"

# Check for WSL
if [ "$OS" = "linux" ] && grep -qi microsoft /proc/version 2>/dev/null; then
    print_info "Windows Subsystem for Linux (WSL) detected"
fi

# Check for required tools
if ! command_exists curl && ! command_exists wget; then
    print_error "Neither curl nor wget found. Please install one:"
    echo "  Ubuntu/Debian: sudo apt-get install curl"
    echo "  CentOS/RHEL:   sudo yum install curl"
    echo "  macOS:         curl is pre-installed"
    exit 1
fi
print_success "Download tools available"

# Check disk space
AVAILABLE_SPACE=$(get_available_disk_space "$HOME")
print_info "Available disk space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt "$MIN_DISK_SPACE_GB" ] 2>/dev/null; then
    print_warning "Low disk space in home! At least ${MIN_DISK_SPACE_GB}GB recommended."
    print_info "Pip cache/tmp will be redirected to project directory automatically."
    if ! prompt_yes_no "Continue anyway?" "y"; then
        exit 1
    fi
fi

# ============================================================================
# Step 1: Conda Setup
# ============================================================================

print_header "Setting Up Conda"

check_conda_health

CONDA_CMD=""
if command_exists mamba; then
    CONDA_CMD="mamba"
    print_success "Mamba found (using for faster installs)"
elif command_exists conda; then
    CONDA_CMD="conda"
    print_success "Conda found"
fi

if command_exists conda; then
    CONDA_PATH=$(which conda)
    print_info "Path: $CONDA_PATH"
    eval "$(conda shell.bash hook 2>/dev/null)" || eval "$(conda shell.zsh hook 2>/dev/null)" || true
else
    if [ "$SKIP_CONDA" = true ]; then
        print_warning "Conda not found, using pip/venv instead"
        if ! command_exists pip && ! command_exists pip3; then
            print_error "pip not found. Please install Python and pip first."
            exit 1
        fi
    else
        print_warning "Conda not found. Installing Miniconda..."

        case "$OS-$ARCH" in
            linux-x86_64)
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                ;;
            linux-aarch64|linux-arm64)
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
                ;;
            macos-arm64)
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
                ;;
            macos-x86_64)
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
                ;;
            *)
                print_error "Automatic conda installation not supported for $OS-$ARCH"
                print_info "Please install Miniconda manually: https://docs.conda.io/en/latest/miniconda.html"
                exit 1
                ;;
        esac

        MINICONDA_PATH="$HOME/miniconda3"
        if [ -d "$MINICONDA_PATH" ]; then
            print_warning "Found existing $MINICONDA_PATH"
            if prompt_yes_no "Try to use existing installation?" "y"; then
                export PATH="$MINICONDA_PATH/bin:$PATH"
                eval "$($MINICONDA_PATH/bin/conda shell.bash hook)" 2>/dev/null || true
                if command_exists conda; then
                    print_success "Using existing conda"
                    CONDA_CMD="conda"
                else
                    print_error "Could not activate. Please fix or remove $MINICONDA_PATH"
                    exit 1
                fi
            else
                exit 1
            fi
        else
            print_info "Downloading Miniconda..."
            TEMP_INSTALLER="/tmp/miniconda_installer_$$.sh"

            if ! download_with_retry "$MINICONDA_URL" "$TEMP_INSTALLER"; then
                print_error "Download failed. Check your internet connection."
                exit 1
            fi

            print_info "Installing Miniconda..."
            bash "$TEMP_INSTALLER" -b -p "$MINICONDA_PATH"
            rm -f "$TEMP_INSTALLER"

            export PATH="$MINICONDA_PATH/bin:$PATH"
            eval "$($MINICONDA_PATH/bin/conda shell.bash hook)"

            # Accept Anaconda TOS (required for newer conda versions)
            print_info "Accepting conda Terms of Service..."
            conda config --set auto_activate_base false
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

            # Configure conda-forge as primary channel (avoids TOS issues)
            print_info "Configuring conda-forge as default channel..."
            conda config --add channels conda-forge
            conda config --set channel_priority strict

            [ -f "$HOME/.bashrc" ] && conda init bash 2>/dev/null || true
            [ -f "$HOME/.zshrc" ] && conda init zsh 2>/dev/null || true

            print_success "Miniconda installed"
            print_warning "Restart your terminal after installation completes"
            CONDA_CMD="conda"
        fi
    fi
fi

# ============================================================================
# Step 2: Environment Creation
# ============================================================================

print_header "Creating Python Environment"

# Accept conda TOS if needed (for existing conda installations)
if [ -n "$CONDA_CMD" ]; then
    # Try to accept TOS silently (only needed for newer conda versions)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

    # Ensure conda-forge is available
    if ! conda config --show channels 2>/dev/null | grep -q conda-forge; then
        conda config --add channels conda-forge 2>/dev/null || true
    fi
fi

if [ "$SKIP_CONDA" = true ] && [ -z "$CONDA_CMD" ]; then
    print_info "Creating virtual environment..."
    VENV_PATH="$SCRIPT_DIR/.venv"

    if [ -d "$VENV_PATH" ]; then
        print_warning "Virtual environment exists at $VENV_PATH"
        if prompt_yes_no "Remove and recreate?" "n"; then
            rm -rf "$VENV_PATH"
        fi
    fi

    [ ! -d "$VENV_PATH" ] && python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"

    # Set explicit paths for venv
    ENV_PIP="$VENV_PATH/bin/pip"
    ENV_PYTHON="$VENV_PATH/bin/python"

    print_success "Virtual environment activated"
else
    if [ -n "$ENV_PREFIX" ]; then
        # Prefix-based conda env (e.g. for project-local envs on shared servers)
        ENV_PREFIX="${ENV_PREFIX/#\~/$HOME}"
        if [[ "$ENV_PREFIX" != /* ]]; then
            ENV_PREFIX="$SCRIPT_DIR/$ENV_PREFIX"
        fi

        if [ -d "$ENV_PREFIX" ]; then
            print_warning "Environment exists at $ENV_PREFIX"
            if prompt_yes_no "Remove and recreate?" "n"; then
                print_info "Removing existing environment..."
                rm -rf "$ENV_PREFIX"
            else
                print_info "Using existing environment"
            fi
        fi

        if [ ! -d "$ENV_PREFIX" ]; then
            print_info "Creating environment at '$ENV_PREFIX' with Python $PYTHON_VERSION..."
            $CONDA_CMD create -p "$ENV_PREFIX" python="$PYTHON_VERSION" -y
        fi

        print_info "Activating environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_PREFIX"
        ENV_PATH="$ENV_PREFIX"
        ENV_PIP="$ENV_PATH/bin/pip"
        ENV_PYTHON="$ENV_PATH/bin/python"
    else
        # Named conda env (default)
        if conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
            print_warning "Environment '$ENV_NAME' already exists"
            if prompt_yes_no "Remove and recreate?" "n"; then
                print_info "Removing existing environment..."
                $CONDA_CMD env remove -n "$ENV_NAME" -y
            else
                print_info "Using existing environment"
            fi
        fi

        if ! conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
            print_info "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
            $CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        fi

        print_info "Activating environment..."
        CONDA_BASE=$(conda info --base)
        ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"

        # IMPORTANT: Use explicit paths to ensure we use the correct environment
        ENV_PIP="$ENV_PATH/bin/pip"
        ENV_PYTHON="$ENV_PATH/bin/python"
    fi
fi

print_success "Environment active"
print_info "Python: $($ENV_PYTHON --version)"

# Always redirect pip cache/tmp to project directory to avoid clutter in home
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRIPT_DIR/.pip_cache}"
PIP_TMP_DIR="${PIP_TMP_DIR:-$SCRIPT_DIR/.pip_tmp}"
mkdir -p "$PIP_CACHE_DIR" "$PIP_TMP_DIR"
export PIP_CACHE_DIR
export TMPDIR="$PIP_TMP_DIR"
print_info "Using pip cache: $PIP_CACHE_DIR"

print_info "Upgrading pip..."
$ENV_PIP install --upgrade pip --quiet

# ============================================================================
# Step 3: PyTorch Installation
# ============================================================================

print_header "Installing PyTorch"

install_pytorch_cpu() {
    print_info "Installing CPU-only PyTorch..."
    $ENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

install_pytorch_cuda() {
    local cuda_version="$1"
    print_info "Installing PyTorch with CUDA $cuda_version..."
    $ENV_PIP install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu${cuda_version}"
}

if [ "$CPU_ONLY" = true ]; then
    install_pytorch_cpu
elif [ "$OS" = "macos" ]; then
    if [ "$ARCH" = "arm64" ]; then
        print_info "Installing PyTorch with MPS support..."
        $ENV_PIP install torch torchvision torchaudio
    else
        install_pytorch_cpu
    fi
else
    if [ "$HAS_NVIDIA_GPU" = true ]; then
        PYTORCH_CUDA="121"

        # Check driver-supported CUDA version via nvidia-smi (critical when nvcc not installed)
        DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
        if [ -n "$DRIVER_CUDA" ]; then
            DRIVER_CUDA_MAJOR=$(echo "$DRIVER_CUDA" | cut -d. -f1)
            DRIVER_CUDA_MINOR=$(echo "$DRIVER_CUDA" | cut -d. -f2)
            if [ "$DRIVER_CUDA_MAJOR" -lt 12 ] 2>/dev/null; then
                PYTORCH_CUDA="118"
                if [ "$DRIVER_CUDA_MAJOR" -eq 11 ] && [ "$DRIVER_CUDA_MINOR" -lt 8 ] 2>/dev/null; then
                    print_warning "Driver supports CUDA $DRIVER_CUDA. Using cu118 wheels (forward-compatible)."
                fi
            fi
        fi

        # Also check nvcc if available (can override)
        if command_exists nvcc; then
            CUDA_TOOLKIT_VERSION=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
            if [ -n "$CUDA_TOOLKIT_VERSION" ]; then
                CUDA_MAJOR=$(echo "$CUDA_TOOLKIT_VERSION" | cut -d. -f1)
                [ "$CUDA_MAJOR" -lt 12 ] 2>/dev/null && PYTORCH_CUDA="118"
            fi
        fi

        install_pytorch_cuda "$PYTORCH_CUDA"
    else
        install_pytorch_cpu
    fi
fi

print_info "Verifying PyTorch..."
if $ENV_PYTHON -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    print_success "PyTorch installed"
    $ENV_PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  MPS (Apple Silicon) available')
else:
    print('  CPU only')
"
else
    print_error "PyTorch verification failed"
    exit 1
fi

# ============================================================================
# Step 4: Dependencies
# ============================================================================

print_header "Installing Dependencies"

REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    # Detect glibc version to cap packages that lack old-glibc wheels
    GLIBC_VERSION=$(ldd --version 2>&1 | head -1 | grep -oP '[0-9]+\.[0-9]+$' || echo "")
    GLIBC_MAJOR=$(echo "$GLIBC_VERSION" | cut -d. -f1)
    GLIBC_MINOR=$(echo "$GLIBC_VERSION" | cut -d. -f2)
    OLD_GLIBC=false
    if [ -n "$GLIBC_MAJOR" ] && [ "$GLIBC_MAJOR" -eq 2 ] 2>/dev/null && [ "$GLIBC_MINOR" -lt 28 ] 2>/dev/null; then
        OLD_GLIBC=true
        print_info "Detected glibc $GLIBC_VERSION (< 2.28); will cap packages to versions with compatible wheels"
    fi

    # Pre-install packages that need version caps on older glibc to avoid source build failures
    if [ "$OLD_GLIBC" = true ]; then
        if grep -E "^av([[:space:]]*[=><~].*)?$" "$REQUIREMENTS_FILE" >/dev/null && ! $ENV_PIP show av 2>/dev/null | grep -q "^Version:"; then
            print_info "Pre-installing av<13 (newer versions lack manylinux_2_17 wheels)..."
            $ENV_PIP install "av<13" --only-binary=av --quiet || true
        fi
        if grep -E "^wandb([[:space:]]*[=><~].*)?$" "$REQUIREMENTS_FILE" >/dev/null && ! $ENV_PIP show wandb 2>/dev/null | grep -q "^Version:"; then
            print_info "Pre-installing wandb<0.19 (newer versions lack manylinux_2_17 wheels)..."
            $ENV_PIP install "wandb<0.19" --quiet || true
        fi
        if grep -E "^h5py([[:space:]]*[=><~].*)?$" "$REQUIREMENTS_FILE" >/dev/null && ! $ENV_PIP show h5py 2>/dev/null | grep -q "^Version:"; then
            print_info "Pre-installing h5py<3.12 (newer versions lack manylinux_2_17 wheels)..."
            $ENV_PIP install "h5py<3.12" --quiet || true
        fi
    fi

    # Step 4a: Install core packages (non-git dependencies) first
    # This ensures basic dependencies are installed even if git packages fail
    # Filter out: torch/torchvision/torchaudio (already installed with correct CUDA index),
    #             decord/decord2 (handled separately), av/wandb/h5py (pre-installed with version caps)
    print_info "Installing core packages..."
    grep -v "@ git+" "$REQUIREMENTS_FILE" | \
        grep -E -v "^(torch|torchvision|torchaudio|av|decord|decord2|wandb|h5py)([[:space:]]*[=><~].*)?$" | \
        $ENV_PIP install -r /dev/stdin && \
        print_success "Core packages installed" || \
        print_warning "Some core packages failed"

    # Step 4b: Install git-based packages one by one (more resilient to failures)
    print_info "Installing git-based packages (this may take a while)..."

    # Ensure git safe.directory allows pip to clone into temp dirs on network volumes
    if ! git config --global --get-all safe.directory 2>/dev/null | grep -q '^\*$'; then
        git config --global --add safe.directory '*'
    fi

    GIT_PACKAGES=$(grep "@ git+" "$REQUIREMENTS_FILE")
    GIT_SUCCESS=0
    GIT_FAILED=0
    FAILED_PACKAGES=""

    while IFS= read -r pkg; do
        [ -z "$pkg" ] && continue
        PKG_NAME=$(echo "$pkg" | cut -d'@' -f1 | tr -d ' ')
        print_info "  Installing $PKG_NAME..."
        if $ENV_PIP install "$pkg" --quiet 2>/dev/null; then
            print_success "    $PKG_NAME installed"
            GIT_SUCCESS=$((GIT_SUCCESS + 1))
        else
            print_warning "    $PKG_NAME failed (non-critical)"
            GIT_FAILED=$((GIT_FAILED + 1))
            FAILED_PACKAGES="$FAILED_PACKAGES $PKG_NAME"
        fi
    done <<< "$GIT_PACKAGES"

    print_info "Git packages: $GIT_SUCCESS succeeded, $GIT_FAILED failed"
    [ -n "$FAILED_PACKAGES" ] && print_warning "Failed packages:$FAILED_PACKAGES"
else
    print_warning "requirements.txt not found"
    print_info "Installing essential packages..."
    $ENV_PIP install numpy scipy scikit-learn pillow opencv-python tqdm h5py transformers timm wandb boto3 gdown google-cloud-storage psutil
fi

# Install decord (platform-specific handling)
print_header "Installing Decord (Video Library)"

install_decord() {
    # Try pip first to avoid conda/pip dependency conflicts (conda pulls conflicting numpy/ffmpeg)
    print_info "Trying pip..."
    if $ENV_PIP install decord 2>/dev/null; then
        print_success "Decord installed via pip"
        return 0
    fi

    # Conda fallback only for named envs (prefix envs mix badly with conda installs)
    if [ -n "$CONDA_CMD" ] && [ "$SKIP_CONDA" != true ] && [ -z "$ENV_PREFIX" ]; then
        print_info "Trying conda-forge..."
        if $CONDA_CMD install -c conda-forge decord -y 2>/dev/null; then
            print_success "Decord installed via conda"
            return 0
        fi
    fi

    # Build from source (needed for macOS ARM64 and some Linux)
    print_info "Pre-built packages not available. Building from source..."

    # Check for required build tools
    if ! command_exists cmake; then
        print_warning "cmake not found, attempting to install..."
        if [ "$OS" = "macos" ]; then
            if command_exists brew; then
                brew install cmake || { print_error "Failed to install cmake"; return 1; }
            else
                print_error "Please install Homebrew first: https://brew.sh"
                print_error "Then run: brew install cmake ffmpeg"
                return 1
            fi
        else
            print_error "Please install cmake: sudo apt-get install cmake"
            return 1
        fi
    fi

    # Install ffmpeg if needed (for development headers)
    if [ "$OS" = "macos" ]; then
        # On macOS, we need ffmpeg from Homebrew for headers
        if ! brew list ffmpeg &>/dev/null; then
            print_info "Installing ffmpeg via Homebrew..."
            if command_exists brew; then
                brew install ffmpeg || { print_error "Failed to install ffmpeg"; return 1; }
            else
                print_error "Please install Homebrew first: https://brew.sh"
                return 1
            fi
        fi
    elif [ "$OS" = "linux" ]; then
        if ! command_exists ffmpeg; then
            print_error "Please install ffmpeg development packages:"
            echo "  Ubuntu/Debian: sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev"
            echo "  CentOS/RHEL: sudo yum install ffmpeg ffmpeg-devel"
            return 1
        fi
    fi

    # Clone and build decord (use TMPDIR when redirected to project dir for low home space)
    local temp_dir="${TMPDIR:-/tmp}/decord_build_$$"
    mkdir -p "$temp_dir"

    print_info "Cloning decord repository..."
    if ! git clone --recursive https://github.com/dmlc/decord.git "$temp_dir/decord"; then
        print_error "Failed to clone decord"
        rm -rf "$temp_dir"
        return 1
    fi

    cd "$temp_dir/decord"

    # Build decord
    print_info "Building decord (this may take a few minutes)..."
    mkdir -p build && cd build

    # Platform-specific cmake configuration
    if [ "$OS" = "macos" ]; then
        # macOS: use Homebrew ffmpeg, disable CUDA
        FFMPEG_DIR=$(brew --prefix ffmpeg 2>/dev/null || echo "/opt/homebrew")
        cmake .. -DCMAKE_BUILD_TYPE=Release \
                 -DUSE_CUDA=OFF \
                 -DFFMPEG_DIR="$FFMPEG_DIR" \
                 -DCMAKE_PREFIX_PATH="$FFMPEG_DIR"
    else
        cmake .. -DCMAKE_BUILD_TYPE=Release
    fi

    # Compile
    local num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    if ! make -j"$num_cores"; then
        print_error "Failed to compile decord"
        cd "$SCRIPT_DIR"
        rm -rf "$temp_dir"
        return 1
    fi

    # Install Python bindings
    cd ../python
    print_info "Installing decord Python bindings..."
    if $ENV_PIP install .; then
        print_success "Decord built and installed successfully"
        cd "$SCRIPT_DIR"
        rm -rf "$temp_dir"
        return 0
    else
        print_error "Failed to install decord Python bindings"
        cd "$SCRIPT_DIR"
        rm -rf "$temp_dir"
        return 1
    fi
}

if ! install_decord; then
    echo ""
    print_warning "Decord installation failed."
    print_info "Video benchmarks may not work without decord."
    echo ""
    echo "To install manually later:"
    if [ "$OS" = "macos" ]; then
        echo "  1. Install build tools: brew install cmake ffmpeg"
        echo "  2. Clone: git clone --recursive https://github.com/dmlc/decord.git"
        echo "  3. Build: cd decord && mkdir build && cd build && cmake .. -DUSE_CUDA=OFF && make"
        echo "  4. Install: cd ../python && pip install ."
    else
        echo "  1. Install build tools: sudo apt-get install cmake ffmpeg libavcodec-dev libavformat-dev libavutil-dev"
        echo "  2. Clone: git clone --recursive https://github.com/dmlc/decord.git"
        echo "  3. Build: cd decord && mkdir build && cd build && cmake .. && make"
        echo "  4. Install: cd ../python && pip install ."
    fi
    echo ""
fi

# ============================================================================
# Step 5: Configuration
# ============================================================================

print_header "Configuring Environment"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/bbscore_data}"
DATA_DIR="${DATA_DIR/#\~/$HOME}"

if ! mkdir -p "$DATA_DIR" 2>/dev/null; then
    print_error "Failed to create: $DATA_DIR"
    exit 1
fi
print_success "Data directory: $DATA_DIR"

DATA_DIR_SPACE=$(get_available_disk_space "$DATA_DIR")
[ "$DATA_DIR_SPACE" -lt 50 ] 2>/dev/null && \
    print_warning "Only ${DATA_DIR_SPACE}GB free. 50GB recommended."

# Detect shell config
detect_shell_config() {
    local shell_name=$(basename "$SHELL")
    case "$shell_name" in
        zsh)  [ -f "$HOME/.zshrc" ] && echo "$HOME/.zshrc" && return ;;
        bash) [ -f "$HOME/.bashrc" ] && echo "$HOME/.bashrc" && return
              [ -f "$HOME/.bash_profile" ] && echo "$HOME/.bash_profile" && return ;;
    esac
    [ -f "$HOME/.bashrc" ] && echo "$HOME/.bashrc" && return
    [ -f "$HOME/.zshrc" ] && echo "$HOME/.zshrc" && return
    [ -f "$HOME/.bash_profile" ] && echo "$HOME/.bash_profile" && return
    echo ""
}

SHELL_CONFIG=$(detect_shell_config)

if [ -n "$SHELL_CONFIG" ]; then
    if ! grep -q "SCIKIT_LEARN_DATA" "$SHELL_CONFIG" 2>/dev/null; then
        cp "$SHELL_CONFIG" "$SHELL_CONFIG.bbscore_backup" 2>/dev/null || true
        {
            echo ""
            echo "# BBScore configuration"
            echo "export SCIKIT_LEARN_DATA=\"$DATA_DIR\""
        } >> "$SHELL_CONFIG"
        print_success "Added SCIKIT_LEARN_DATA to $SHELL_CONFIG"
    else
        print_info "SCIKIT_LEARN_DATA already configured"
    fi
else
    print_warning "Could not detect shell config"
    echo "  Add manually: export SCIKIT_LEARN_DATA=\"$DATA_DIR\""
fi

export SCIKIT_LEARN_DATA="$DATA_DIR"

# Create server-specific activation script (leave upstream activate_bbscore.sh untouched)
HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "local")
ACTIVATE_SUFFIX="${ACTIVATE_SUFFIX:-$HOSTNAME_SHORT}"
ACTIVATE_SCRIPT="$SCRIPT_DIR/activate_bbscore_${ACTIVATE_SUFFIX}.sh"

cat > "$ACTIVATE_SCRIPT" << 'SCRIPT_HEADER'
#!/bin/bash
# Activate BBScore environment (server-specific)
# Usage: source activate_bbscore_SUFFIX.sh

SCRIPT_HEADER

if [ -n "$CONDA_CMD" ] && [ "$SKIP_CONDA" != true ]; then
    if [ -n "$ENV_PREFIX" ]; then
        cat >> "$ACTIVATE_SCRIPT" << EOF
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="$ENV_PATH"
if command -v conda &> /dev/null; then
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "\$ENV_PATH"
else
    echo "Error: conda not found"
    return 1
fi
EOF
    else
        cat >> "$ACTIVATE_SCRIPT" << EOF
if command -v conda &> /dev/null; then
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
else
    echo "Error: conda not found"
    return 1
fi
EOF
    fi
else
    cat >> "$ACTIVATE_SCRIPT" << 'EOF'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
EOF
fi

cat >> "$ACTIVATE_SCRIPT" << 'ACTIVATE_TAIL'

export SCIKIT_LEARN_DATA="DATA_DIR_PLACEHOLDER"

# GPU selection: default to GPU 0 to reserve larger GPUs for heavy tasks
# Override before sourcing: export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Disable torch.compile/inductor on older drivers (avoids Triton PTX errors)
export TORCHDYNAMO_DISABLE=1

PYTHON_PATH=$(which python)
echo "BBScore environment activated!"
echo "Python: $PYTHON_PATH"
echo "Data: $SCIKIT_LEARN_DATA"
echo ""
echo "Quick start:"
echo "  $PYTHON_PATH run.py --model resnet18 --layer layer4 --benchmark OnlineTVSDV1 --metric ridge"
ACTIVATE_TAIL

# Replace placeholders with actual values
sed -i.bak "s|DATA_DIR_PLACEHOLDER|$DATA_DIR|g" "$ACTIVATE_SCRIPT" && rm -f "$ACTIVATE_SCRIPT.bak"

chmod +x "$ACTIVATE_SCRIPT"
print_success "Created $(basename "$ACTIVATE_SCRIPT")"

# ============================================================================
# Step 6: Verify Key Imports
# ============================================================================

print_header "Verifying Installation"

print_info "Checking critical imports..."

# Check core packages
IMPORT_ERRORS=""
for pkg in torch numpy scipy sklearn PIL cv2 tqdm h5py transformers timm wandb psutil; do
    if $ENV_PYTHON -c "import $pkg" 2>/dev/null; then
        print_success "  $pkg"
    else
        print_warning "  $pkg - NOT FOUND"
        IMPORT_ERRORS="$IMPORT_ERRORS $pkg"
    fi
done

# Check jepa (critical for some benchmarks)
print_info "Checking jepa..."
if $ENV_PYTHON -c "from jepa.src.models.attentive_pooler import AttentiveClassifier" 2>/dev/null; then
    print_success "  jepa.src.models.attentive_pooler"
else
    print_warning "  jepa import failed"
    print_info "  You may need to reinstall jepa:"
    echo "    $ENV_PIP uninstall jepa -y"
    echo "    $ENV_PIP install git+https://github.com/thekej/jepa.git"
    IMPORT_ERRORS="$IMPORT_ERRORS jepa"
fi

if [ -n "$IMPORT_ERRORS" ]; then
    print_warning "Missing packages:$IMPORT_ERRORS"
    print_info "You can install missing packages manually with:"
    echo "  $ENV_PIP install <package_name>"
fi

# ============================================================================
# Step 7: System Check
# ============================================================================

print_header "System Check"

CHECK_SCRIPT="$SCRIPT_DIR/check_system.py"
if [ -f "$CHECK_SCRIPT" ]; then
    print_info "Running diagnostics..."
    $ENV_PYTHON "$CHECK_SCRIPT" --quick 2>/dev/null && print_success "System check passed" || \
        print_warning "System check reported issues (non-critical)"
else
    print_warning "check_system.py not found, skipping"
fi

# ============================================================================
# Step 8: Cleanup temp dirs
# ============================================================================

if [ -d "$SCRIPT_DIR/.pip_tmp" ]; then
    rm -rf "$SCRIPT_DIR/.pip_tmp"
    print_success "Cleaned up pip temp directory"
fi

if [ "$CLEAN_CACHE" = true ] && [ -d "$SCRIPT_DIR/.pip_cache" ]; then
    rm -rf "$SCRIPT_DIR/.pip_cache"
    print_success "Cleaned up pip cache (--clean)"
fi

# ============================================================================
# Done!
# ============================================================================

print_header "Installation Complete!"

# Summary of any issues
if [ -n "$IMPORT_ERRORS" ]; then
    echo -e "${YELLOW}Note: Some packages failed to install:${NC}$IMPORT_ERRORS"
    echo "These may be optional. Check if your benchmarks need them."
    echo ""
fi

# Use the explicit environment Python path
PYTHON_PATH="$ENV_PYTHON"

echo -e "${GREEN}${BOLD}BBScore is ready to use!${NC}"
echo ""
echo "To get started:"
echo ""
echo -e "  ${CYAN}1.${NC} Activate the environment:"
echo -e "     ${GREEN}source activate_bbscore_${ACTIVATE_SUFFIX}.sh${NC}"
[ -n "$CONDA_CMD" ] && [ "$SKIP_CONDA" != true ] && [ -z "$ENV_PREFIX" ] && \
    echo -e "     or: ${GREEN}conda activate $ENV_NAME${NC}"
echo ""
echo -e "  ${CYAN}2.${NC} Run a quick test:"
echo -e "     ${GREEN}${PYTHON_PATH} run.py --model resnet18 --layer layer4 --benchmark OnlineTVSDV1 --metric ridge${NC}"
echo ""
echo -e "  ${CYAN}3.${NC} Check your system:"
echo -e "     ${GREEN}${PYTHON_PATH} check_system.py${NC}"
echo ""
echo "Python: $PYTHON_PATH"
echo "Data directory: $DATA_DIR"
echo ""

[ -n "$SHELL_CONFIG" ] && print_warning "Restart your terminal or: source $SHELL_CONFIG"

echo ""
print_success "Happy benchmarking! 🧠"
