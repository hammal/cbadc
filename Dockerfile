# cbadc with ngspice: the image the circuit bench (tests/ngspice) runs in.
#
# Debian trixie ships ngspice 44 built with XSPICE, which the bench needs for
# its digital clock, flip-flops, bridges and trnoise sources, so nothing is
# compiled here. The same image is the dev container.
#
#   docker build -t cbadc-ngspice .
#   docker run --rm cbadc-ngspice                      # the bench
#   docker run --rm cbadc-ngspice ./scripts/test-unit  # anything else
FROM python:3.15-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends ngspice git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /cbadc
ENV MPLBACKEND=Agg UV_LINK_MODE=copy
# Dependencies first so that editing the source does not re-resolve them.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --extra dev --extra delsig
COPY . .

CMD ["./scripts/test-performance", "ngspice"]
