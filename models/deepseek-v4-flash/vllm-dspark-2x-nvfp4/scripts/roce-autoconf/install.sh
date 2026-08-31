#!/usr/bin/env bash
# 在一台 DGX Spark 上安装光口自适应（root）：
#   ./install.sh 10.0.0.2 10.0.0.3    # head
#   ./install.sh 10.0.0.3 10.0.0.2    # worker
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOCAL="${1:?本机 RoCE IP}"; PEER="${2:?对端 RoCE IP}"
install -m 755 "$HERE/roce-autoconf.sh" /usr/local/sbin/roce-autoconf.sh
echo "LOCAL_IP=$LOCAL PEER_IP=$PEER" > /etc/ds4-roce.conf
install -m 644 "$HERE/roce-autoconf.service" /etc/systemd/system/roce-autoconf.service
systemctl daemon-reload && systemctl enable roce-autoconf.service
/usr/local/sbin/roce-autoconf.sh
