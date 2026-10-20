# Egress proxy: the agent container's only way out. Forwards the model APIs, refuses and
# logs everything else (DESIGN.md §7). Configured at start by entrypoint.sh.
FROM alpine:3.20
RUN apk add --no-cache tinyproxy
COPY allow.txt /etc/tinyproxy/allow.txt
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
