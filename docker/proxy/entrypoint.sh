#!/bin/sh
# The harness runs this container on the host network and passes the one address to
# listen on (the run's internal-network gateway) and the one subnet to accept.
set -eu
: "${FSB_PROXY_LISTEN:?}" "${FSB_PROXY_ALLOW:?}" "${FSB_PROXY_PORT:=8888}"

cat > /tmp/tinyproxy.conf <<EOF
User nobody
Group nobody
Port ${FSB_PROXY_PORT}
Listen ${FSB_PROXY_LISTEN}
Allow ${FSB_PROXY_ALLOW}
Timeout 600
MaxClients 50
# Info logs every connection and every refusal; the log is kept with the run.
LogLevel Info
ConnectPort 443
FilterDefaultDeny Yes
Filter "/etc/tinyproxy/allow.txt"
EOF

exec tinyproxy -d -c /tmp/tinyproxy.conf
