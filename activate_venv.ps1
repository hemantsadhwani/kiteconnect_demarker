# PowerShell Script - DO NOT RUN DIRECTLY FROM BASH
# ====================================================
# This is a PowerShell script (.ps1). It cannot be executed directly from bash.
#
# If you are using bash/zsh on Linux/macOS:
#   Use: ./activate_venv.sh  or  source ./activate_venv.sh
#
# If you have PowerShell installed and want to use this script:
#   Use: pwsh -File ./activate_venv.ps1
#   Or in PowerShell: . .\activate_venv.ps1
#
# To install PowerShell on Linux:
#   sudo dnf install powershell -y  (Amazon Linux 2023)
#   sudo apt install powershell     (Ubuntu/Debian)
# ====================================================

# Cross-platform Virtual Environment Activation Script for PowerShell
# Works on Windows PowerShell, PowerShell Core (Linux/macOS), and Windows
#
# IMPORTANT: This script MUST be dot-sourced to work properly!
# Usage: . .\activate_venv.ps1
#        (Note the dot-space before the script name)
#
# For Linux/macOS bash: use ./activate_venv.sh instead

# Check if we're actually running in PowerShell
if (-not $PSVersionTable) {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: This script requires PowerShell!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "You are trying to run a PowerShell script from bash." -ForegroundColor Yellow
    Write-Host "Please use the bash script instead:" -ForegroundColor Yellow
    Write-Host "  ./activate_venv.sh" -ForegroundColor Cyan
    Write-Host "  or" -ForegroundColor Yellow
    Write-Host "  source ./activate_venv.sh" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "If you have PowerShell installed, run:" -ForegroundColor Yellow
    Write-Host "  pwsh -File ./activate_venv.ps1" -ForegroundColor Cyan
    exit 1
}

# Check if script is being dot-sourced (required for environment changes to persist)
# When dot-sourced, $MyInvocation.InvocationName will be '.' or the script name
# When executed directly, it will be the full path
$isDotSourced = $MyInvocation.InvocationName -eq '.' -or 
                ($MyInvocation.InvocationName -ne $null -and 
                 $MyInvocation.InvocationName -eq (Split-Path -Leaf $MyInvocation.MyCommand.Path))

if (-not $isDotSourced) {
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "IMPORTANT: Script must be dot-sourced!" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To activate the virtual environment, run:" -ForegroundColor Cyan
    Write-Host "  . .\activate_venv.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Note the dot-space ('. ') before the script name!" -ForegroundColor Yellow
    Write-Host "This is required for environment changes to persist in your current shell." -ForegroundColor Yellow
    Write-Host ""
    # Determine OS for alternative activation path
    $isWindows = $PSVersionTable.Platform -eq 'Win32NT' -or $IsWindows -or ($PSVersionTable.PSVersion.Major -le 5)
    if ($isWindows) {
        Write-Host "Alternatively, you can activate directly with:" -ForegroundColor Cyan
        Write-Host "  . .\venv\Scripts\Activate.ps1" -ForegroundColor White
    } else {
        Write-Host "Alternatively, you can activate directly with:" -ForegroundColor Cyan
        Write-Host "  . .\venv\bin\activate" -ForegroundColor White
        Write-Host "  (Note: This requires bash, not PowerShell)" -ForegroundColor Yellow
    }
    Write-Host ""
    exit 1
}

# Set execution policy for current process (if needed)
try {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -ErrorAction SilentlyContinue
} catch {
    # Policy might already be set, continue
}

# Function to find venv directory
function Find-Venv {
    $currentDir = Get-Location
    
    # Determine path separator and activation script location based on OS
    $isWindows = $PSVersionTable.Platform -eq 'Win32NT' -or $IsWindows -or ($PSVersionTable.PSVersion.Major -le 5)
    
    if ($isWindows) {
        $scriptsDir = "Scripts"
        $activateScript = "Activate.ps1"
    } else {
        # Linux/macOS with PowerShell Core
        $scriptsDir = "bin"
        $activateScript = "activate"
    }
    
    # Check current directory
    $venvPath = Join-Path $currentDir "venv"
    $activatePath = Join-Path $venvPath (Join-Path $scriptsDir $activateScript)
    if (Test-Path $activatePath) {
        return $venvPath
    }
    
    $venvPath = Join-Path $currentDir ".venv"
    $activatePath = Join-Path $venvPath (Join-Path $scriptsDir $activateScript)
    if (Test-Path $activatePath) {
        return $venvPath
    }
    
    # Check parent directory
    $parentDir = Split-Path -Parent $currentDir
    $venvPath = Join-Path $parentDir "venv"
    $activatePath = Join-Path $venvPath (Join-Path $scriptsDir $activateScript)
    if (Test-Path $activatePath) {
        return $venvPath
    }
    
    $venvPath = Join-Path $parentDir ".venv"
    $activatePath = Join-Path $venvPath (Join-Path $scriptsDir $activateScript)
    if (Test-Path $activatePath) {
        return $venvPath
    }
    
    return $null
}

# Find virtual environment
$venvPath = Find-Venv

if ($null -eq $venvPath) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please ensure you have created a virtual environment named 'venv' or '.venv'" -ForegroundColor Yellow
    Write-Host "You can create one with: python -m venv venv" -ForegroundColor Yellow
    return
}

# Determine activation script path based on OS
$isWindows = $PSVersionTable.Platform -eq 'Win32NT' -or $IsWindows -or ($PSVersionTable.PSVersion.Major -le 5)

if ($isWindows) {
    $activateScript = Join-Path $venvPath (Join-Path "Scripts" "Activate.ps1")
} else {
    # Linux/macOS with PowerShell Core - use bash activation script
    $activateScript = Join-Path $venvPath (Join-Path "bin" "activate")
}

if (Test-Path $activateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    
    if ($isWindows) {
        # Windows: Dot-source the PowerShell activation script
        . $activateScript
    } else {
        # Linux/macOS: Manually set environment variables (PowerShell can't source bash scripts)
        # Set VIRTUAL_ENV
        $env:VIRTUAL_ENV = $venvPath
        
        # Update PATH to prepend venv/bin
        $venvBin = Join-Path $venvPath "bin"
        $pathSeparator = [System.IO.Path]::PathSeparator
        $pathArray = $env:PATH -split $pathSeparator
        
        # Remove venv/bin if already in PATH (to avoid duplicates)
        $pathArray = $pathArray | Where-Object { $_ -ne $venvBin }
        
        # Prepend venv/bin to PATH
        $env:PATH = $venvBin + $pathSeparator + ($pathArray -join $pathSeparator)
        
        # Note: PowerShell prompt will show (venv) if VIRTUAL_ENV is set
        # (depends on PowerShell prompt customization)
    }
    
    # Verify activation
    if ($env:VIRTUAL_ENV) {
        Write-Host "Virtual environment activated successfully!" -ForegroundColor Green
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCmd) {
            $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
        }
        if ($pythonCmd) {
            Write-Host "Python: $($pythonCmd.Source)" -ForegroundColor Green
        }
        Write-Host "Virtual Env: $env:VIRTUAL_ENV" -ForegroundColor Green
    } else {
        Write-Host "Warning: Virtual environment may not have activated correctly." -ForegroundColor Yellow
        Write-Host "Try running: . $activateScript" -ForegroundColor Yellow
    }
} else {
    Write-Host "Error: Activation script not found at $activateScript" -ForegroundColor Red
    return
}
