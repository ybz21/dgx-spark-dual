#!/usr/bin/env bash
# 把运维层装到两台上：光口自适应 + 看门狗 + 开机自启。
# 在 head 上跑一次即可，它会把 worker 侧也装好。
# 幂等，可重复执行。
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
CFG=$here/glm53-ops.env
. "$CFG"
WORKER_HOST=${WORKER_HOST:-glm53-worker.local}
peer(){ ssh -o StrictHostKeyChecking=no "$SSH_USER@$WORKER_HOST" "$@"; }

for required in "$HOME/launch-glm53-vllm-tp2.sh" "$HOME/glm53-warmup.sh"; do
  [ -x "$required" ] || { echo "缺少可执行依赖: $required" >&2; exit 1; }
done
peer 'test -x ~/launch-glm53-vllm-tp2.sh' \
  || { echo "worker 缺少 ~/launch-glm53-vllm-tp2.sh" >&2; exit 1; }

echo "== head: 装脚本和配置 =="
sudo install -m 0644 "$CFG"                 /etc/glm53-ops.env
sudo install -m 0755 "$here/glm53-rail.sh"      /usr/local/sbin/glm53-rail.sh
sudo install -m 0755 "$here/glm53-supervise.sh" /usr/local/bin/glm53-supervise.sh
sudo install -m 0644 "$here/systemd/glm53-rail.service"       /etc/systemd/system/
sudo install -m 0644 "$here/systemd/glm53-rail.timer"         /etc/systemd/system/
sudo install -m 0644 "$here/systemd/glm53-supervisor.service" /etc/systemd/system/
# supervisor 要能免密调 rail 脚本重新探测光口
echo "$SSH_USER ALL=(root) NOPASSWD: /usr/local/sbin/glm53-rail.sh" \
  | sudo tee /etc/sudoers.d/glm53-rail >/dev/null
sudo chmod 0440 /etc/sudoers.d/glm53-rail
sudo touch /var/log/glm53-supervisor.log && sudo chown "$SSH_USER" /var/log/glm53-supervisor.log
sudo systemctl daemon-reload
sudo systemctl enable --now glm53-rail.service
sudo systemctl enable --now glm53-rail.timer

echo "== worker: 装脚本和配置 =="
tar -C "$here" -cf - glm53-ops.env glm53-rail.sh systemd/glm53-rail.service systemd/glm53-rail.timer systemd/glm53-worker-boot.service \
  | peer 'mkdir -p ~/glm53-ops && tar -C ~/glm53-ops -xf - && sed -i "s/^NODE_ROLE=.*/NODE_ROLE=slave/" ~/glm53-ops/glm53-ops.env'
peer 'sudo install -m 0644 ~/glm53-ops/glm53-ops.env /etc/glm53-ops.env
      sudo install -m 0755 ~/glm53-ops/glm53-rail.sh /usr/local/sbin/glm53-rail.sh
      sudo install -m 0644 ~/glm53-ops/systemd/glm53-rail.service        /etc/systemd/system/
      sudo install -m 0644 ~/glm53-ops/systemd/glm53-rail.timer          /etc/systemd/system/
      sudo install -m 0644 ~/glm53-ops/systemd/glm53-worker-boot.service /etc/systemd/system/
      sudo systemctl daemon-reload
      sudo systemctl enable --now glm53-rail.service
      sudo systemctl enable --now glm53-rail.timer
      sudo systemctl enable glm53-worker-boot.service'

echo "== head: 启用看门狗 =="
sudo systemctl enable --now glm53-supervisor.service

echo
echo "装好了。查看:"
echo "  systemctl status glm53-supervisor --no-pager"
echo "  tail -f /var/log/glm53-supervisor.log"
echo "  cat /run/glm53-rail.env        # 探测到的光口"
