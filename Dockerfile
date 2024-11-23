FROM rust:1.79 as builder

WORKDIR /usr/src/docnote
COPY . .
ENV SQLX_OFFLINE=true
RUN cargo build --release

FROM debian:bookworm

RUN apt update && apt upgrade && apt install -y openssl ca-certificates

COPY --from=builder /usr/src/docnote/target/release/docnote /usr/local/bin/docnote

RUN update-ca-certificates

CMD ["docnote"]