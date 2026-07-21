#!/usr/bin/env bash
set -euo pipefail

# ROS installations often expose third-party pytest plugins through PYTHONPATH.
# Disable automatic plugin discovery so this project runs only its own tests.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python -m pytest -q
