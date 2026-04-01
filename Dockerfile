FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Base dependencies + Python + C++
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    curl wget gnupg2 ca-certificates \
    build-essential g++ \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS + TypeScript (multiple-js, multiple-ts)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g typescript \
    && rm -rf /var/lib/apt/lists/*

# Java + Scala (multiple-java, multiple-scala)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk scala \
    && rm -rf /var/lib/apt/lists/*

# Go (multiple-go)
RUN apt-get update && apt-get install -y --no-install-recommends golang-go \
    && rm -rf /var/lib/apt/lists/*

# Ruby, PHP, Lua, R (multiple-rb, multiple-php, multiple-lua, multiple-r)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ruby php-cli lua5.4 r-base \
    && ln -sf /usr/bin/lua5.4 /usr/bin/lua \
    && rm -rf /var/lib/apt/lists/*

# Mono C# (multiple-cs) — uses csc + mono
RUN apt-get update && apt-get install -y --no-install-recommends \
    mono-mcs mono-runtime \
    && ln -sf /usr/bin/mcs /usr/bin/csc \
    && rm -rf /var/lib/apt/lists/*

# Racket (multiple-rkt)
RUN apt-get update && apt-get install -y --no-install-recommends racket \
    && rm -rf /var/lib/apt/lists/*

# Rust (multiple-rs)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /app

WORKDIR /app

RUN test -f /app/generations.json && rm /app/generations.json || true

RUN pip3 install .

# DS-1000 test_code.py files use the deprecated `parser` module (removed in 3.12,
# also missing from some 3.10 rebuilds). Install a minimal compatibility shim.
RUN cp /app/parser_shim.py "$(python3 -c 'import site; print(site.getsitepackages()[0])')/parser.py"

RUN mkdir -p /workspace/results /workspace/logs

EXPOSE 8094

CMD ["python3", "/app/api/main.py"]
