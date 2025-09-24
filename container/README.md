# IndexTTS Container Usage

This guide is for **advanced users** who are familiar with container technologies. The container is compatible with both **Docker** and **Podman**.


## Initial Setup and Configuration

To get started, you'll need to create a `compose.yml` file to configure the project for your system. We've provided a template, `compose.example.yml`, with all the necessary settings. Simply make a copy of this file and rename it to `compose.yml`.

When you have your `compose.yml` file, you can safely edit it to fit your needs. Your local changes will be ignored by Git, ensuring that any adjustments you make won't cause conflicts when you pull future updates.

We recommend that you carefully review the file, as it contains all the configuration details for the container, including GPU settings, port mappings, and environment variables. You may need to adapt these settings to fit your specific system.

- **GPU Support**:
  - The example configuration is set up to automatically use all **NVIDIA** GPUs. This *requires* the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** to be installed and **[correctly configured](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration)** on your host system. You do **not** need the full CUDA Toolkit installed on your host, but a **modern NVIDIA driver** is essential.
  - If you have an **AMD GPU**, you'll need to manually add its device paths to the `devices` section, since AMD's toolkit doesn't support auto-mounting yet.
  - If you are not using a GPU, you should remove the `devices` section entirely.
  - If you are using Docker Desktop for **Windows**, you *must* [follow their instructions to get GPU support](https://docs.docker.com/desktop/features/gpu/) on Windows hosts.
- **SELinux**: SELinux is disabled by default to allow GPU access. If you have custom SELinux policies that permit access, then you can remove the `security_opt` section for improved container isolation.
- **Host User/Group IDs**: The container is configured to run as the host user's IDs to ensure correct permissions for mapped directories. The default `1000:1000` is correct for most Linux users. You can check your IDs by running `id` in your host terminal.
- **Ports**: The container exposes port `7860` for the WebUI. You can change this, or restrict access to `localhost` by uncommenting the alternative port mapping instead.
- **Environment Variables**: The `environment` section controls app behavior, including dependency installation and WebUI launch parameters. Adjust these values as needed.
- **Package Mirrors**: If you experience slow downloads from the default PyPI index, you can configure an alternative mirror via the `UV_DEFAULT_INDEX` environment variable.


## Running the Container

After your `compose.yml` is configured, you can use the standard `docker compose` or `podman compose` commands to build and run the container.


### Build and Run

To build the container image and run the service in the background, use this command:

```bash
docker compose up -d

podman compose up -d
```

> [!TIP]
> To view the logs in real-time in your terminal, remove the `-d` flag to run the service in the foreground. If the container is already running, the command will simply attach to the existing service and display its live logs.


### Stopping the Container

To stop the running container, use one of these commands to cleanly shut down and unload the virtual network:

```bash
docker compose down

podman compose down
```


## Running Custom Commands

The `compose.yml` file specifies a default `command` to run on startup (`webui`). However, you can easily override this to run different commands without modifying the `compose.yml` file.

To run a specific command, use the `run` subcommand with your chosen service name (`indextts`):

```bash
# Example: Run the GPU diagnostic tool.
docker compose run --service-ports indextts gpu-check

podman compose run --service-ports indextts gpu-check
```

Some of the available built-in commands are:

  - `webui`: Launches the IndexTTS WebUI (the default).
  - `gpu-check`: A diagnostic tool to detect available hardware accelerators.
  - `shell`: Starts an interactive shell inside the container. Useful during development, since it allows you to quickly run and restart custom code inside the container. If your container was started as a service, run `docker attach indextts` in another terminal to connect to the shell.
  - `update`: Updates the application to the latest version of the code via Git.
  - You can even run your own custom commands or scripts, such as `uv run your_app.py`.

> [!WARNING]
> The custom command will run in a new, temporary container, with the same mounted volumes and resources as the normal service. However, it will be isolated from other containers on your system (such as the main service).
> 
> The `--service-ports` flag tells the new, temporary container to listen to all service ports, so that network features inside the container (such as Gradio) become reachable from the host network. However, this means that your temporary container will fail to start if those ports are already bound to something else. You can remove that flag if you don't require networking in your temporary container.
>
> Lastly, the act of running temporary containers may leave "orphan containers" on your system. You can clean them up by adding the `--remove-orphans` flag to any of the `compose` sub-commands (such as `compose down --remove-orphans` or `compose run --remove-orphans`).


## Additional Information

- **App Updates**: The application code is mounted as a volume. This allows for rapid development and app updates without needing to rebuild the container image. To update the application, you can either use the `update` command mentioned above, or navigate to the source directory on your host machine and run these commands manually:

```bash
git pull
git lfs pull
```

- **Troubleshooting**: If you encounter issues, the service container is configured with `stdin_open: true` and `tty: true`. This allows you to attach directly to the **live runtime environment** of the active service container, to view logs or open an interactive shell for debugging. To start an interactive Bash session in your running service container, use the `docker exec` or `podman exec` commands:

```bash
docker exec -it indextts bash

podman exec -it indextts bash
```

- **Single Instance**: The `container_name: indextts` setting enforces a custom name, which is great for managing a single instance, but must be removed if you plan to run multiple instances on the same host.

- **Rebuilding the Container Image**: Under normal conditions, you don't need to rebuild the container image itself, since it's designed to be a lightweight operating runtime for the app. However, if you've updated the `Containerfile`, you can force a manual rebuild using one of these commands:

```bash
docker compose build

podman compose build
```

- **"Just" Support**: For a more streamlined experience, install "[just](https://github.com/casey/just)" to use the convenient commands provided in our `justfile`.

-----

**Disclaimer**

This container is provided as a convenience "as-is" and is intended for advanced users. While the `compose.yml` offers a solid starting point, users are expected to adapt the configuration to suit their specific system environment and needs.
