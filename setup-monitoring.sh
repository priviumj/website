#!/bin/bash

# Setup script for Privium website monitoring
# This configures a cron job to run the monitor every 10 minutes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor-site.js"
NODE_PATH=$(which node)

echo "Setting up Privium website monitoring..."
echo "Script location: $MONITOR_SCRIPT"
echo "Node path: $NODE_PATH"

# Check if node is installed
if [ -z "$NODE_PATH" ]; then
    echo "Error: Node.js is not installed or not in PATH"
    exit 1
fi

# Make sure the monitor script is executable
chmod +x "$MONITOR_SCRIPT"

# Create the cron job entry
CRON_ENTRY="*/10 * * * * $NODE_PATH $MONITOR_SCRIPT >> $SCRIPT_DIR/monitor-cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$MONITOR_SCRIPT"; then
    echo "Cron job already exists. Removing old entry..."
    crontab -l 2>/dev/null | grep -v "$MONITOR_SCRIPT" | crontab -
fi

# Add the cron job
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo ""
echo "✓ Monitoring setup complete!"
echo ""
echo "The monitor will run every 10 minutes and check:"
echo "  - Home page (privium.com.au)"
echo "  - Services page"
echo "  - About page"
echo "  - Contact page"
echo "  - Client Portal login"
echo ""
echo "Logs will be saved to:"
echo "  - monitor-log.json (JSON format with results)"
echo "  - monitor-cron.log (cron execution log)"
echo ""
echo "To test the monitor manually, run:"
echo "  node $MONITOR_SCRIPT"
echo ""
echo "To view current cron jobs:"
echo "  crontab -l"
echo ""
echo "To remove the monitoring cron job:"
echo "  crontab -l | grep -v monitor-site.js | crontab -"
echo ""
