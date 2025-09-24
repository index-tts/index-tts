#!/usr/bin/env bash

set -eu


# Verify that the `/app` data volume is actually mounted.
if [[ ! -d "/app/container" ]]; then
    echo "Error: \"/app\" volume is not mounted. Exiting..." >&2
    exit 1
fi


# Verify that the HOME volume is actually mounted.
if [[ ! -d "${HOME}" ]]; then
    echo "Error: Home volume \"${HOME}\" is not mounted. Exiting..." >&2
    exit 1
fi


# Verify that launch commands have been provided and aren't empty.
if [[ "$#" -lt 1 || -z "$1" ]]; then
    echo "Error: No launch command provided." >&2
    exit 1
fi


# Utility functions.
function escape_shell_command {
    local escaped_cmd
    # NOTE: "%q" ensures shell-compatible argument escaping.
    printf -v escaped_cmd " %q" "$@"
    # Remove the leading space from the escaped command.
    printf "%s" "${escaped_cmd# }"
}

function print_command {
    # NOTE: "%s" prints the escaped command as-is without parsing escape-seqs.
    printf "[IndexTTS] + %s\n" "$(escape_shell_command "$@")"
}


# Executes a command and displays the exact command for logging purposes.
# NOTE: The current script continues running and will resume execution afterwards.
function run_cmd {
    print_command "$@"
    "$@"
}


# Terminates the current shell and replaces its PID with the given command.
# NOTE: Required in containers to ensure the final process receives all signals,
# so that graceful shutdown commands from the container host will actually work.
function exec_cmd {
    print_command "$@"
    exec "$@"
}


# Execute the HOME shell launch script, since it won't run by default due to how
# we're changing the HOME location via environment overrides.
# NOTE: This is necessary because we can't override `/etc/passwd` in the container,
# since the host's UID/GID that we run as may be an anonymous user in the container.
if [[ -f "${HOME}/.bashrc" ]]; then
    source "${HOME}/.bashrc"
fi


# Disable third-party analytics.
# SEE: https://www.gradio.app/guides/sharing-your-app#analytics
export GRADIO_ANALYTICS_ENABLED="False"
# SEE: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubdisabletelemetry
export HF_HUB_DISABLE_TELEMETRY=1


# Add the `uv tool` binary install directory to PATH, so that tools installed
# via `uv tool install` can be executed directly without `uv tool` afterwards.
uv_tool_bin_dir="$(uv tool dir --bin)"
if [[ -z "${uv_tool_bin_dir}" ]]; then
    echo "Error: Failed to detect uv tool binary directory." >&2
    exit 1
fi
if [[ ":${PATH}:" != *":${uv_tool_bin_dir}:"* ]]; then
    echo "Adding uv tool installation directory to PATH..."
    run_cmd uv tool update-shell
    export PATH="${uv_tool_bin_dir}:${PATH}"
fi


# Don't use `uv` hardlinks, since we're inside a container with mounted volumes.
# NOTE: Using plain symlinks also ensures that our `container_home` is portable
# to other disks (or other machines) without data becoming duplicated.
# WARNING: Don't run `uv cache clean`, since the symlinks depend on the cache.
# SEE: https://docs.astral.sh/uv/reference/cli/#uv-run--link-mode
export UV_LINK_MODE="symlink"


# Optimize startup of container by compiling Python source files to bytecode.
# SEE: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
export UV_COMPILE_BYTECODE=1


# Use a different env directory inside the container (if not overridden already).
# SEE: https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
# NOTE: Ensures that the host's environment won't conflict with the container's.
# NOTE: Relative paths are always relative to the project directory.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv_container}"


# Ensure that all container scripts are executable.
# NOTE: We ignore the entrypoint itself, since it's built into the container
# image and is owned by root, so we can't modify it (and we don't need to).
find /opt/scripts/actions -name "*.sh" -exec chmod +x {} \;


# Ensure that we are in the correct directory.
cd /app


# Force installation of WebUI dependencies if user requested "webui" command.
[[ "$1" == "webui" ]] && INSTALL_WEBUI="yes"


# Install/verify that the `uv` env is up-to-date and contains all dependencies.
# NOTE: This also removes previous extra dependencies unless we request the
# same `--extra` flags every time here. Be aware of that. However, `uv` is very
# fast and will link/unlink them on-demand in a fraction of a second.
# NOTE: We always install the extra WebUI dependencies if that env var is missing,
# for convenience if people run the Containerfile directly without `compose.yml`.
# NOTE: To use a custom mirror, set the `UV_DEFAULT_INDEX` env for the container.
# NOTE: We don't perform any syncing if the user has requested an app update.
if [[ "$1" != "update" ]]; then
    echo "Verifying dependencies..."
    sync_args=()
    [[ "${INSTALL_WEBUI:-}" != "no" ]] && sync_args+=("--extra" "webui")
    [[ "${INSTALL_DEEPSPEED:-}" == "yes" ]] && sync_args+=("--extra" "deepspeed")
    run_cmd uv sync "${sync_args[@]}"
fi


# Determine the correct command.
_cmd=()
if [[ "$1" == "webui" ]]; then
    echo "Starting IndexTTS WebUI..."
    
    webui_args=()
    [[ -n "${WEBUI_PORT:-}" ]] && webui_args+=("--port" "${WEBUI_PORT}")
    [[ "${WEBUI_FP16:-}" == "yes" ]] && webui_args+=("--fp16")
    [[ "${WEBUI_DEEPSPEED:-}" == "yes" ]] && webui_args+=("--deepspeed")
    [[ "${WEBUI_CUDA_KERNEL:-}" == "yes" ]] && webui_args+=("--cuda_kernel")
    [[ -n "${WEBUI_SEG_TOKENS:-}" ]] && webui_args+=("--gui_seg_tokens" "${WEBUI_SEG_TOKENS}")
    
    _cmd=(uv run webui.py "${webui_args[@]}")
elif [[ "$1" == "gpu-check" ]]; then
    echo "Running GPU check..."
    _cmd=(uv run tools/gpu_check.py)
elif [[ "$1" == "shell" ]]; then
    echo "Running interactive shell..."
    _cmd=(bash)
elif [[ "$1" == "update" ]]; then
    echo "Updating application..."
    _cmd=(/opt/scripts/actions/update.sh)
else
    echo "Running custom command..."
    _cmd=("$@")
fi


# Execute and replace this process (PID 1) with the final command.
# NOTE: This is *critically important* for container signal handling.
exec_cmd "${_cmd[@]}"
