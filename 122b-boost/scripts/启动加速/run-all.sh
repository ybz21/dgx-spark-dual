#!/bin/bash
# 总驱动：顺序跑 A → B → C，每个之间清理，最后输出对比表
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

START=$(date +%s)
log "=== run-all PoC: A (runai) → B (shard) → C (custom) ==="

bash "$SCRIPT_DIR/A-runai.sh" 2>&1 | tee "$RESULTS_DIR/A.console.log"
bash "$SCRIPT_DIR/B-shard.sh" 2>&1 | tee "$RESULTS_DIR/B.console.log"
bash "$SCRIPT_DIR/C-custom.sh" 2>&1 | tee "$RESULTS_DIR/C.console.log"

END=$(date +%s)
log "=== All done in $((END - START))s ==="

# 汇总
echo
echo "================== SUMMARY =================="
printf "%-8s %-12s %-8s %-10s %s\n" SCHEME BOOT_TIME OOM HEALTHY ERROR
printf "%-8s %-12s %-8s %-10s %s\n" ------ --------- --- ------- -----
for s in A B C; do
    f="$RESULTS_DIR/$s/result.json"
    if [ -f "$f" ]; then
        python3 -c "
import json,sys
r = json.load(open('$f'))
print(f\"{r['scheme']:<8} {str(r.get('boot_time_s','-')):<12} {str(r['oom']).lower():<8} {str(r['healthy']).lower():<10} {r.get('error') or '-'}\")
"
    else
        printf "%-8s %s\n" "$s" "(no result)"
    fi
done
echo
echo "Detailed logs:  $RESULTS_DIR/{A,B,C}/{boot,dump,tensorize}.log"
echo "Recommendation: pick the scheme with smallest BOOT_TIME and OOM=false"
