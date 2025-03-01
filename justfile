jup:
  uv run --with jupyter jupyter lab

@lint:
  uv run ruff check src

@fmt:
  uv run ruff format src

@run $file:
  uv run python $file


