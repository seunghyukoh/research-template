# AI Research Template

## Why you should use this template

1. **Battery included**
    - Start your research with minimal setup—just configure the `.env` file, and you’re ready to go.
2. **Structured and Scalable**
    - A well-organized directory structure clarifies the purpose of each directory.
    - Designed for minimal changes when adding new experiments, making it easier to scale your project.
3. **Easy Environment Setup on New Servers**
    - The template includes tools and scripts to simplify the process of setting up a consistent environment on any new server.
4. **Collaboration-Ready**
    - Equipped with tools and configurations for seamless team collaboration:
        - **W&B and Huggingface Plugins**: Easily integrate tracking and model sharing services into your workflow.
        - **Code Style Enforcement with Ruff**: Maintain a clean and consistent codebase by adhering to predefined coding conventions.
        - **Pre-Commit Hooks**: Ensure high-quality commits by enforcing rules and checks before code is pushed.

## Before you start

1. **Install recommended extensions**
    - I strongly recommend you to use VSCode.
    - Go to the `Extensions` tab in the left sidebar and install recommended extensions

        ```text
        - Better Commits
        - Editor Config
        - Error Lens
        - Pre-commit helper
        - Python environment manager
        - SFTP
        - Ruff Formatter
        ```

2. **Setup the environment configuration file:**
    - Duplicate `.env.example` and rename it to `.env`.
    - Edit the `.env` file to include your specific settings:

        ```txt
        PROJECT_NAME=research-template

        WANDB_API_KEY=
        WANDB_USERNAME=
        WANDB_PROJECT=research-template
        # Options: `checkpoint`, `end`, `false`
        WANDB_LOG_MODEL=end
        ```

3. **Setup the environment:**
    - Side effect: A Python package manager [`uv`](https://docs.astral.sh/uv/) will be installed.
    - Run the following command to configure your environment:

        ```bash
        bash setup-environment.sh
        ```

4. **(Optional) Configure SFTP settings:**
    - Duplicate `.vscode/sftp.example.json` and rename it to `.vscode/sftp.json`.
    - Edit the new `sftp.json` file to include your SFTP settings:

        ```json
        {
            "name": "Name",
            "protocol": "sftp",
            "openSsh": false,
            "port": 22,
            "host": "HostName",
            "remotePath": "/remote/path",
            "username": "UserName",
            "uploadOnSave": true,
            "useTempFile": true,
            "ignoreFile": ".sftpignore",
            "sshConfigPath": "/path/to/ssh_config",
            "syncOption": {
                "delete": true,
                "update": true
            }
        }
        ```

---

## How to use this template

1. Python Files and Directory Structure
    - **Primary Location**: All Python files should reside in the `src/` directory.
    - **Purpose of `src/`**: For writing code related to experiments and pre/post-processing.
    - **Shared Code**: Use the `src/packages/` subdirectory for reusable modules shared across different scripts.

2. Experiments and Configurations
    - Store experimental functions and configurations in the `experiments/` directory.
    - **Organization**: Create subdirectories named after each experiment to avoid file clutter.
    - **YAML Configs**: Use YAML files for storing arguments or configurations for experiments.
    - **Advantages**: Facilitates repeating experiments or sharing configurations across multiple experiments.

3. Utility Scripts
    - The `utils/` directory is for scripts that enhance productivity or manage repetitive tasks.

    **Notable Utilities**
    1. ``wait-for-pid.sh``: Waits for a process to finish before running another command.
        - How to Use:

            ```bash
            bash utils/wait-for-pid.sh 193345 && bash whatever-u-want.sh
            ```

    2. ``remote-debug.sh``: Enables remote debugging via VS Code by using SFTP to synchronize files with a remote server.
        - How to Use:
            1. Start the Python file in debugging mode on the server:

                ```bash
                source utils/remote-debug.sh

                debug whatever-u-want.py --args args1
                ```

            2. Open VS Code’s Debugger (Shift + Cmd + D).
                - Select the “Python Debugger: Remote” configuration from .vscode/launch.json.
                - Ensure that the server’s port 5678 is forwarded to the local port 5678.

---

## Useful Links

- **Weights & Biases Setup Guide:**
  Learn how to set up and use Weights & Biases for your machine learning projects.
  [Weights & Biases Quickstart Documentation](https://docs.wandb.ai/quickstart)
