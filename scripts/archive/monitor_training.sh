#!/bin/bash
# Monitor PAT-L training progress

echo "🔍 PAT-L Training Monitor"
echo "========================"
echo ""

# Check if tmux session exists
if tmux has-session -t pat-gpu 2>/dev/null; then
    echo "✅ Training session 'pat-gpu' is running"
    echo ""
    echo "📊 Latest training output:"
    echo "------------------------"
    tmux capture-pane -t pat-gpu -p | tail -30 | grep -E "Epoch|loss|AUC|Stage|━|🔵|🟡|🟢"
    echo ""
else
    echo "❌ No training session found"
    echo "Start training with: ./scripts/launch_pat_training_gpu.sh"
    exit 1
fi

# Check GPU usage
echo "🖥️  GPU Status:"
echo "-------------"
nvidia-smi --query-gpu=gpu_name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{
    printf "GPU: %s\n", $1
    printf "Temp: %s°C | GPU Util: %s%% | Mem Util: %s%%\n", $2, $3, $4
    printf "Memory: %s MB / %s MB\n", $5, $6
}'

echo ""
echo "📁 Log files:"
ls -lht logs/pat_training/*.log 2>/dev/null | head -3

echo ""
echo "💡 Commands:"
echo "  tmux attach -t pat-gpu    # Watch live training"
echo "  tmux kill-session -t pat-gpu  # Stop training"
echo "  tail -f logs/pat_training/*.log  # View logs"