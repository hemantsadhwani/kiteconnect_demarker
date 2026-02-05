#!/bin/bash

# Script to configure SSH keepalive on EC2 instance
# This helps prevent "Connection reset" errors

echo "Configuring SSH keepalive settings..."

# Backup original config
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup.$(date +%Y%m%d_%H%M%S)

# Configure ClientAliveInterval (send keepalive every 60 seconds)
if grep -q "^ClientAliveInterval" /etc/ssh/sshd_config; then
    sudo sed -i 's/^ClientAliveInterval.*/ClientAliveInterval 60/' /etc/ssh/sshd_config
else
    echo "ClientAliveInterval 60" | sudo tee -a /etc/ssh/sshd_config
fi

# Configure ClientAliveCountMax (disconnect after 3 failed attempts = 3 minutes)
if grep -q "^ClientAliveCountMax" /etc/ssh/sshd_config; then
    sudo sed -i 's/^ClientAliveCountMax.*/ClientAliveCountMax 3/' /etc/ssh/sshd_config
else
    echo "ClientAliveCountMax 3" | sudo tee -a /etc/ssh/sshd_config
fi

# Enable TCPKeepAlive
if grep -q "^TCPKeepAlive" /etc/ssh/sshd_config; then
    sudo sed -i 's/^TCPKeepAlive.*/TCPKeepAlive yes/' /etc/ssh/sshd_config
else
    echo "TCPKeepAlive yes" | sudo tee -a /etc/ssh/sshd_config
fi

# Remove commented lines if they exist
sudo sed -i 's/^#ClientAliveInterval.*/ClientAliveInterval 60/' /etc/ssh/sshd_config
sudo sed -i 's/^#ClientAliveCountMax.*/ClientAliveCountMax 3/' /etc/ssh/sshd_config
sudo sed -i 's/^#TCPKeepAlive.*/TCPKeepAlive yes/' /etc/ssh/sshd_config

# Test SSH config
echo "Testing SSH configuration..."
sudo sshd -t

if [ $? -eq 0 ]; then
    echo "✓ SSH configuration is valid"
    echo "Restarting SSH service..."
    sudo systemctl restart sshd
    echo "✓ SSH service restarted successfully"
    echo ""
    echo "Configuration applied:"
    echo "  - ClientAliveInterval: 60 seconds"
    echo "  - ClientAliveCountMax: 3 attempts"
    echo "  - TCPKeepAlive: enabled"
    echo ""
    echo "Note: You may need to reconnect your SSH session for changes to take effect."
else
    echo "✗ SSH configuration test failed. Restoring backup..."
    sudo cp /etc/ssh/sshd_config.backup.* /etc/ssh/sshd_config
    echo "Backup restored. Please check the configuration manually."
    exit 1
fi

