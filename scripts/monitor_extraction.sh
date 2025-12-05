#!/bin/bash
# Monitor feature extraction log for coverage_norm errors

LOG_DIR="artifacts/20251130"
ERROR_KEYWORDS=("coverage_norm" "NameError" "Traceback")

echo "Monitoring feature extraction log..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    LATEST_LOG=$(ls -t ${LOG_DIR}/full_extraction_*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        # Check for errors
        for keyword in "${ERROR_KEYWORDS[@]}"; do
            if grep -q "$keyword" "$LATEST_LOG" 2>/dev/null; then
                echo "=========================================="
                echo "ERROR DETECTED: $keyword"
                echo "Time: $(date)"
                echo "=========================================="
                grep -A 5 -B 5 "$keyword" "$LATEST_LOG" | tail -n 20
                echo ""
            fi
        done
        
        # Show last few lines
        echo "[$(date +%H:%M:%S)] Last log lines:"
        tail -n 3 "$LATEST_LOG" 2>/dev/null | sed 's/^/  /'
        echo ""
    else
        echo "[$(date +%H:%M:%S)] Waiting for log file..."
    fi
    
    sleep 30
done




