#!/usr/bin/env bash
# 光口自适应：插上哪个 QSFP 口都能自动认出来并配好地址。
#
# 为什么不能只看 carrier：实机上两对口都插了缆（enp1s0f1np1 和 enP2p1s0f1np1
# carrier 都是 1），只有一对真正连着对端。所以判据是"配上地址后 ping 得通对端"，
# 不是"有没有光"。
#
# 幂等：已经配好且能通就直接返回，不动网络。
# 两台跑同一份脚本，按管理网 IP 自认角色。
set -u
CFG=${CFG:-/etc/glm53-ops.env}
[ -r "$CFG" ] && . "$CFG"
: "${RAIL_CANDIDATES:=enp1s0f1np1 enp1s0f0np0 enP2p1s0f1np1 enP2p1s0f0np0}"
: "${RAIL_PREFIX:=24}"
log() { echo "[rail] $*"; }

# --- 自认角色 -------------------------------------------------------------
ROLE=${ROLE:-${NODE_ROLE:-}}
case "$ROLE" in
  master|head) ROLE=head; MY_IP=$HEAD_RAIL_IP; PEER_IP=$WORKER_RAIL_IP ;;
  slave|worker) ROLE=worker; MY_IP=$WORKER_RAIL_IP; PEER_IP=$HEAD_RAIL_IP ;;
  *) log "无法确认本机角色（管理网 IP 尚未就绪或配置错误）"; exit 2 ;;
esac
log "role=$ROLE  自己=$MY_IP  对端=$PEER_IP"

hca_of() {  # 网口名 -> RoCE HCA 名
  local ifname=$1 d
  for d in /sys/class/infiniband/*/device/net/"$ifname"; do
    [ -e "$d" ] && basename "$(dirname "$(dirname "$(dirname "$d")")")" && return 0
  done
  return 1
}

emit() {   # 把结果写给 launcher 和 supervisor 用
  local ifname=$1 hca=$2
  local out=${RAIL_OUT:-/run/glm53-rail.env}
  { echo "RAIL_IF=$ifname"; echo "RAIL_HCA=$hca"; echo "RAIL_ROLE=$ROLE"
    echo "RAIL_MY_IP=$MY_IP"; echo "RAIL_PEER_IP=$PEER_IP"; } > "$out"
  log "已就绪: if=$ifname hca=$hca -> $out"
}

# --- 1) 当前配置就能通 -> 什么都不做 --------------------------------------
cur=$(ip -o -4 addr show | awk -v ip="$MY_IP/" '$4 ~ "^"ip {print $2; exit}')
if [ -n "$cur" ] && ping -c1 -W2 -I "$cur" "$PEER_IP" >/dev/null 2>&1; then
  log "已配好且对端可达，不动网络"
  emit "$cur" "$(hca_of "$cur" || echo unknown)"; exit 0
fi

# --- 2) 逐个候选口试：配地址 -> ping 对端 ---------------------------------
# 对端可能还没起来（同时开机），所以整轮重试若干次。
for round in 1 2 3 4 5 6; do
  for ifname in $RAIL_CANDIDATES; do
    [ -e "/sys/class/net/$ifname" ] || continue
    [ "$(cat "/sys/class/net/$ifname/carrier" 2>/dev/null)" = 1 ] || continue
    # 别动已经配着别的业务地址的口
    if ! ip -o -4 addr show "$ifname" | awk '{print $4}' | grep -q "^$MY_IP/"; then
      ip link set "$ifname" up 2>/dev/null
      ip addr add "$MY_IP/$RAIL_PREFIX" dev "$ifname" 2>/dev/null
      added=1
    else added=0; fi
    if ping -c1 -W2 -I "$ifname" "$PEER_IP" >/dev/null 2>&1; then
      log "第 $round 轮: $ifname 通了"
      emit "$ifname" "$(hca_of "$ifname" || echo unknown)"; exit 0
    fi
    # 这个口不通，把刚加的地址撤掉再试下一个，免得多个口挂同一地址
    [ "$added" = 1 ] && ip addr del "$MY_IP/$RAIL_PREFIX" dev "$ifname" 2>/dev/null
  done
  log "第 $round 轮没通，等对端…"; sleep 10
done

# --- 3) 全部试完仍不通：保底配在第一个有光的口上 --------------------------
# 对端可能只是开机慢。配上地址让 supervisor 后面自己重试，不要在这里死掉。
for ifname in $RAIL_CANDIDATES; do
  [ "$(cat "/sys/class/net/$ifname/carrier" 2>/dev/null)" = 1 ] || continue
  ip link set "$ifname" up 2>/dev/null
  ip addr add "$MY_IP/$RAIL_PREFIX" dev "$ifname" 2>/dev/null
  log "对端暂时不可达，保底配在 $ifname，交给 supervisor 重试"
  emit "$ifname" "$(hca_of "$ifname" || echo unknown)"; exit 0
done
log "没有任何光口有 carrier —— 检查光缆"; exit 1
