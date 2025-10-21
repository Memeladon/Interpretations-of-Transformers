FROM ghcr.io/astral-sh/uv:0.5.17-python3.13-bookworm

WORKDIR /

COPY public/ public/
COPY src/ src/
COPY migration/ migration/
COPY alembic.ini .
COPY docker.env .env

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy from the cache instead of linking since it's a mounted volume
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the FastAPI application by default
# Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# Uses `--host 0.0.0.0` to allow access from outside the container
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn src.start:app --host 0.0.0.0 --port 5000 --proxy-headers --forwarded-allow-ips=*"]
