#!/bin/bash
# Run two commands in parallel using tmux

SESSION_NAME="server_session"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session with first command
tmux new-session -d -s $SESSION_NAME -c "$PWD"

# Run first command in the first pane
tmux send-keys -t $SESSION_NAME "cd openpi && mamba activate openpi && python scripts/serve_policy.py --env LIBERO" C-m

# Split window horizontally and run second command
tmux split-window -h -t $SESSION_NAME -c "$PWD"
tmux send-keys -t $SESSION_NAME "cd large-video-planner && mamba activate ei_world_model && python server.py" C-m

# Attach to the session
tmux attach -t $SESSION_NAME
