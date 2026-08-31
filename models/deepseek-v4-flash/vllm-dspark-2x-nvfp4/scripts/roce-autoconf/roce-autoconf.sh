#!/usr/bin/env bash
# roce-autoconf —— DGX Spark 双机直连光口自适应
#
# 每台机器有两块 ConnectX-7（共 4 个 QSFP 口）。光纤随便插哪个口，本脚本在开机/手动
# 运行时自动找到"插了线且对端可达"的那个口，把本机的 RoCE 点对点 IP 固定上去
# （NetworkManager 静态连接 roce-tp，持久化），并把 NCCL 需要的三个参数
# （NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX）写进：
#   - /etc/ds4-roce.env               （通用真值，别处 source 用）
#   - ~ai/ds4-dspark-2x/.env.dspark   （存在时就地更新）
#
# 安装（每台一次，root）：
#   install -m 755 roce-autoconf.sh /usr/local/sbin/roce-autoconf.sh
#   echo "LOCAL_IP=10.0.0.2 PEER_IP=10.0.0.3" > /etc/ds4-roce.conf   # head；worker 反过来
#   然后装 systemd 单元（见文末注释）或直接跑一次。
#
# 选口逻辑：
#   候选 = 有 rdma 设备（CX7）且 carrier=1 的网口；
#   已带管理网 IPv4（如 DHCP 拿到 192.168.x）的口后置——那是插到交换机上的；
#   逐个候选：绑 IP → ping 对端（每口最多 PROBE_SEC 秒）→ 通了就定下来；
#   全部不通则整轮重试（对端可能还在开机），RETRY_ROUNDS 轮后停在第一候选上，
#   保证至少有 IP 在网（对端起来后再跑一次即可收敛，或对端跑它自己的本脚本）。
set -uo pipefail

CONF=/etc/ds4-roce.conf
ENV_OUT=/etc/ds4-roce.env
DSPARK_ENV="${DSPARK_ENV:-/home/ai/ds4-dspark-2x/.env.dspark}"
PROBE_SEC="${PROBE_SEC:-6}"
RETRY_ROUNDS="${RETRY_ROUNDS:-20}"
RETRY_SLEEP="${RETRY_SLEEP:-10}"

log(){ echo "[roce-autoconf $(date +%T)] $*"; }
die(){ log "✗ $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "需要 root"
[ -f "$CONF" ] || die "缺 $CONF（内容形如 LOCAL_IP=10.0.0.2 PEER_IP=10.0.0.3）"
# shellcheck disable=SC1090
. <(tr ' ' '\n' < "$CONF")
: "${LOCAL_IP:?}" "${PEER_IP:?}"

hca_of(){ ls "/sys/class/net/$1/device/infiniband" 2>/dev/null | head -1; }

candidates(){
  local pri="" sec=""
  for d in /sys/class/net/*; do
    i="$(basename "$d")"
    [ -n "$(hca_of "$i")" ] || continue
    [ "$(cat "$d/carrier" 2>/dev/null)" = 1 ] || continue
    # 已从别处（DHCP/手工）拿了非本链路 IPv4 的口，多半插在交换机上，后置
    if ip -4 -br addr show "$i" | grep -qE "[0-9]+\.[0-9]+" && ! ip -4 -br addr show "$i" | grep -q "$LOCAL_IP"; then
      sec="$sec $i"
    else
      pri="$pri $i"
    fi
  done
  echo $pri $sec
}

bind_ip(){ # $1=ifname —— 用 NM 静态连接固化（无 NM 时退回 ip 命令）
  if command -v nmcli >/dev/null && systemctl is-active -q NetworkManager 2>/dev/null; then
    nmcli con delete roce-tp >/dev/null 2>&1
    nmcli con add type ethernet ifname "$1" con-name roce-tp \
      ipv4.method manual ipv4.addresses "$LOCAL_IP/24" ipv6.method link-local \
      connection.autoconnect yes >/dev/null && nmcli con up roce-tp >/dev/null 2>&1
  else
    ip addr replace "$LOCAL_IP/24" dev "$1"; ip link set "$1" up
  fi
}

gid_index(){ # $1=hca —— 找 RoCE v2 + 本机 IPv4 映射的 GID index
  local hca="$1" want="::ffff:$LOCAL_IP"
  for g in "/sys/class/infiniband/$hca/ports/1/gids/"*; do
    idx="$(basename "$g")"
    gid="$(cat "$g" 2>/dev/null)" || continue
    typ="$(cat "/sys/class/infiniband/$hca/ports/1/gid_attrs/types/$idx" 2>/dev/null)" || continue
    case "$gid" in *"${LOCAL_IP//./:}"|*ffff:*) ;; *) continue;; esac
    python3 - "$gid" "$want" <<'PY' || continue
import ipaddress, sys
a = ipaddress.ip_address(sys.argv[1]); b = ipaddress.ip_address(sys.argv[2])
sys.exit(0 if a == b else 1)
PY
    [ "$typ" = "RoCE v2" ] && { echo "$idx"; return 0; }
  done
  echo 3   # 探测不到时用 CX7 惯例值
}

settle(){ # $1=ifname —— 写 env 真值 + 更新 .env.dspark
  local ifname="$1" hca; hca="$(hca_of "$ifname")"
  local gid; gid="$(gid_index "$hca")"
  printf 'ROCE_IFNAME=%s\nROCE_HCA=%s\nROCE_GID_INDEX=%s\nROCE_LOCAL_IP=%s\nROCE_PEER_IP=%s\n' \
    "$ifname" "$hca" "$gid" "$LOCAL_IP" "$PEER_IP" > "$ENV_OUT"
  log "定口：$ifname (HCA=$hca GID=$gid) $LOCAL_IP ↔ $PEER_IP，已写 $ENV_OUT"
  if [ -f "$DSPARK_ENV" ]; then
    sed -i -E "s/^NCCL_SOCKET_IFNAME=.*/NCCL_SOCKET_IFNAME=$ifname/; s/^NCCL_IB_HCA=.*/NCCL_IB_HCA=$hca/; s/^NCCL_IB_GID_INDEX=.*/NCCL_IB_GID_INDEX=$gid/" "$DSPARK_ENV"
    log "已同步 $DSPARK_ENV 的 NCCL_SOCKET_IFNAME/NCCL_IB_HCA/NCCL_IB_GID_INDEX"
  fi
}

# 幂等快路径：当前已有口带着 LOCAL_IP 且对端可达 → 只刷新 env 输出
cur="$(ip -4 -br addr | awk -v ip="$LOCAL_IP" '$0 ~ ip"/" {print $1}' | head -1)"
if [ -n "$cur" ] && ping -c1 -W1 -I "$cur" "$PEER_IP" >/dev/null 2>&1; then
  log "现状已通（$cur），不动网络"; settle "$cur"; exit 0
fi

round=0
while [ "$round" -lt "$RETRY_ROUNDS" ]; do
  round=$((round+1))
  cands="$(candidates)"
  [ -n "$cands" ] || { log "第 $round 轮：没有带光的 CX7 口，等 $RETRY_SLEEP s"; sleep "$RETRY_SLEEP"; continue; }
  log "第 $round 轮候选：$cands"
  for i in $cands; do
    bind_ip "$i"
    end=$(( $(date +%s) + PROBE_SEC ))
    while [ "$(date +%s)" -lt "$end" ]; do
      if ping -c1 -W1 -I "$i" "$PEER_IP" >/dev/null 2>&1; then settle "$i"; exit 0; fi
      sleep 1
    done
    log "  $i 上 ping $PEER_IP 不通，换下一个口"
  done
  sleep "$RETRY_SLEEP"
done

first="$(candidates | awk '{print $1}')"
if [ -n "$first" ]; then
  bind_ip "$first"; settle "$first"
  log "⚠ 对端始终不可达，先停在 $first 上（对端起来后任一端重跑本脚本即可收敛）"
  exit 0
fi
die "没有任何可用的 CX7 光口"

# systemd 单元（安装脚本会写 /etc/systemd/system/roce-autoconf.service）：
#   [Unit]
#   Description=RoCE point-to-point autoconf (DGX Spark x2)
#   After=network-online.target NetworkManager.service
#   Wants=network-online.target
#   [Service]
#   Type=oneshot
#   ExecStart=/usr/local/sbin/roce-autoconf.sh
#   TimeoutStartSec=0
#   [Install]
#   WantedBy=multi-user.target
