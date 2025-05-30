#!/bin/bash

# FunTinyBrain Automated Visual Monitor
# Runs the Go program 100 times with automatic progression

# Configuration
RUNS=10
PATTERN_SWITCH_INTERVAL=50  # Should match your Go code

# Setup temp files
TEMPFILE=$(mktemp)
SPIKE_FILE=$(mktemp)
SUMMARY_FILE=$(mktemp)

# Cleanup function
cleanup() {
    rm -f "$TEMPFILE" "$SPIKE_FILE" "$SUMMARY_FILE"
}
trap cleanup EXIT

# Simple ASCII bar graph function
graph_spikes() {
    local spikes=$1
    local pattern=$2
    local step=$3
    
    # Create bar
    local bar=""
    for ((i=0; i<spikes; i++)); do
        bar+="■"
    done
    
    # Color coding
    if [ "$pattern" = "A" ]; then
        echo -e "\e[34mStep $step (Pattern A): $bar $spikes spikes\e[0m"
    else
        echo -e "\e[31mStep $step (Pattern B): $bar $spikes spikes\e[0m"
    fi
}

# Main monitoring function
monitor_run() {
    local run_num=$1
    echo -e "\n\e[1mRun $run_num/$RUNS\e[0m" | tee -a "$SUMMARY_FILE"
    
    go run main.go | tee "$TEMPFILE" | while read -r line; do
        if [[ "$line" == *"Output:"* ]]; then
            # Extract pattern and spikes
            step=$(echo "$line" | awk '{print $2}' | tr -d ':')
            pattern=$(echo "$line" | awk '{print $4}' | tr -d '[]')
            output=$(echo "$line" | awk '{print $NF}' | tr -d '[]')
            
            # Count spikes (1s)
            spikes=$(echo "$output" | tr ' ' '\n' | grep -c "1")
            
            # Store for summary
            echo "$step $pattern $spikes" >> "$SPIKE_FILE"
            
            # Show live graph (last 10 steps)
            clear
            echo "=== Live Spike Monitor (Run $run_num) ===" | tee -a "$SUMMARY_FILE"
            tail -n 10 "$SPIKE_FILE" | while read -r data; do
                s=$(echo "$data" | awk '{print $1}')
                p=$(echo "$data" | awk '{print $2}')
                sp=$(echo "$data" | awk '{print $3}')
                graph_spikes "$sp" "$p" "$s"
            done | tee -a "$SUMMARY_FILE"
        fi
    done
    
    # Store compact summary
    echo -e "\n=== Run $run_num Compact Summary ===" >> "$SUMMARY_FILE"
    awk '{a[$2]+=$3; count[$2]++} END {print "Pattern A:",a["A"]/count["A"],"avg spikes"; print "Pattern B:",a["B"]/count["B"],"avg spikes"}' "$SPIKE_FILE" >> "$SUMMARY_FILE"
    
    # Reset for next run
    > "$SPIKE_FILE"
}

# Main execution
clear
echo "Starting automated experiment with $RUNS runs..."
for ((run=1; run<=RUNS; run++)); do
    monitor_run "$run"
    
    # Show progress
    echo -e "\nCompleted run $run/$RUNS"
    if [ "$run" -lt "$RUNS" ]; then
        echo "Proceeding to next run in 1 second..."
        sleep 1
    fi
done

# Final summary
clear
echo -e "\nExperiment complete! Final summary:\n"
cat "$SUMMARY_FILE"

echo -e "\nDetailed data available in:"
echo "Raw output: $TEMPFILE"
echo "Spike data: $SPIKE_FILE"
echo "Full summary: $SUMMARY_FILE"