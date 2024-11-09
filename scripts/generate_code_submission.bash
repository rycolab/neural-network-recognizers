set -euo pipefail

output_file=code.zip
rm -f "$output_file"
zip -r "$output_file" . \
  --include \
    'Dockerfile*' \
    '.docker*' \
    'poetry.lock' \
    'pyproject.toml' \
    'README.md' \
    'src/*' \
    'experiments/*' \
    'data/*' \
    'tests/*' \
    'scripts/*' \
  --exclude \
    '*/.git*' \
    '*/__pycache__/*' \
    '*.pyc' \
    '*.swp' \
    '*/.pytest_cache/*' \
    '*/.mypy_cache/*'
