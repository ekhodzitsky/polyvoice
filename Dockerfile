# Build stage
FROM ubuntu:24.04 AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.96.0 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /usr/src/polyvoice
COPY . .
RUN cargo build --release --features cli --bin polyvoice

# Runtime stage
FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 65532 --create-home --home-dir /home/polyvoice \
        --shell /usr/sbin/nologin polyvoice
COPY --from=builder /usr/src/polyvoice/target/release/polyvoice /usr/local/bin/polyvoice
USER polyvoice
ENTRYPOINT ["polyvoice"]
CMD ["--help"]
