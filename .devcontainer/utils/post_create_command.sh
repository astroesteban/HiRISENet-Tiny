#!/bin/bash
set -e
###
# File Name:
#   post_create_command.sh
#
# Details:
#   This script is used to run commands after the Docker development container
#   is created.
###

# `````````````````````````````````````````````````````````````````````````````
# Function name: _help()
#
# Description:
#   Provides help information
#
function _help() {
    echo -e "Usage: $0 [arguments]"
    echo "Options:"
    echo "    -h | --help) Print usage information"
    echo "    --user-packages) Installs the user's python packages"
    echo "    --nice-bash) Sets up a nices CLI bash prompt"
}

###############################################################################
################################ ENTRYPOINT ###################################
###############################################################################

ORIGIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PYTHON_VENV_DIRECTORY=/tmp/picovenv/bin/activate
INSTALL_USER_PACKAGES=OFF
SETUP_NICE_BASH=OFF

# Get the CLI options
for i in "$@"; do
  case $i in
    --fprime-packages)
      INSTALL_FPRIME_PACKAGES=ON
      shift
      ;;
    --user-packages)
      INSTALL_USER_PACKAGES=ON
      shift
      ;;
    --nice-bash)
      SETUP_NICE_BASH=ON
      shift
      ;;
    -h|--help)
      _help
      exit 0
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;

    *)
      ;;
  esac
done

# Install the user's package dependencies
if [ ${INSTALL_USER_PACKAGES} = ON ]; then
  echo ">>> Installing the user packages"
  conda env create --file=/workspace/environment.yml
  conda init
fi

# Setup a nicer bash prompt
if [ ${SETUP_NICE_BASH} = ON ]; then
    echo ">>> Setting up a nicer bash experience"
    cat ${ORIGIN_DIR}/setup_bash_prompt.sh >> ~/.bashrc
fi
