#!/usr/bin/env bash

set -e


# ** DON'T MODIFY THIS BOOTSTRAP ENTRYPOINT FILE! **

# Run the launch logic within this process (no subshell) by sourcing it directly.
# NOTE: This indirection is required for Windows host compatibility, since NTFS
# doesn't preserve "+x" executable bits when we mount newer scripts as a volume.
# This basic, executable "entrypoint.sh" is just a bootstrap to run the real code.

source /opt/scripts/actions/launch.sh
