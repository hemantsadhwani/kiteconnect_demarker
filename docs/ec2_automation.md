# EC2 automation (start/stop scheduling)

This doc explains how to **automatically start and stop an EC2 instance at specific times**.

## Important concept (read this first)

- **Inside-the-instance automation (cron/systemd)** can *stop* the instance, but it **cannot start itself again** (because it’s off).
- For **true start + stop automation**, use something **outside the instance**, such as:
  - **EventBridge Scheduler + Lambda** (recommended), or
  - **A separate always-on machine**, like your **Windows PC** via Task Scheduler.

Also note:

- If you **stop/start** the instance, its **public IP changes** unless you attach an **Elastic IP**.
- If the instance is in an **Auto Scaling Group**, stopping it may cause it to be replaced.

## Option A (recommended): EventBridge Scheduler + Lambda (works even when instance is OFF)

High level:

- Create **two schedules**: one to **Start**, one to **Stop**
- Each schedule triggers a **Lambda** function
- Lambda calls `ec2:StartInstances` / `ec2:StopInstances` for your instance ID(s)

### Why this is best

- Runs fully in AWS (no laptop/PC needs to be on)
- Supports timezone-aware scheduling
- Easy to extend to multiple instances

### Minimal Lambda example (Python)

Create one Lambda and deploy it twice (or one Lambda with an `ACTION` env var).

```python
import os
import boto3

ec2 = boto3.client("ec2")

INSTANCE_IDS = [i.strip() for i in os.environ["INSTANCE_IDS"].split(",") if i.strip()]
ACTION = os.environ["ACTION"].strip().lower()  # "start" or "stop"

def handler(event, context):
    if ACTION == "start":
        return ec2.start_instances(InstanceIds=INSTANCE_IDS)
    if ACTION == "stop":
        return ec2.stop_instances(InstanceIds=INSTANCE_IDS)
    raise ValueError(f"Unsupported ACTION={ACTION!r}")
```

### Minimal IAM policy for the Lambda role

Replace region/account/instance-id.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "StartStopSpecificInstance",
      "Effect": "Allow",
      "Action": [
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ec2:DescribeInstances"
      ],
      "Resource": "arn:aws:ec2:REGION:ACCOUNT_ID:instance/i-xxxxxxxxxxxxxxxxx"
    }
  ]
}
```

### Scheduling examples

EventBridge Scheduler supports cron + timezone. Example patterns:

- **Weekdays start 09:00**: `cron(0 9 ? * MON-FRI *)`
- **Weekdays stop 15:35**: `cron(35 15 ? * MON-FRI *)`

If you’re also using **cron inside the instance** (see `docs/cron_usage.md`), start the instance **a few minutes earlier** than your in-instance cron time; cron does not “catch up” for missed times during boot.

## Option B: Windows Task Scheduler + AWS CLI (works if your Windows PC is ON)

Yes—you can schedule EC2 start/stop from your **local Windows machine**.

### When to choose this

- You don’t want to set up Lambda/EventBridge yet
- Your Windows PC is **reliably on** at those times (or it’s a server/workstation that stays on)

### Prerequisites

- Install **AWS CLI v2** on Windows
- Configure credentials (least privilege recommended):

```powershell
aws configure --profile ec2-scheduler
```

This writes to `%UserProfile%\.aws\credentials` and `%UserProfile%\.aws\config`.

### PowerShell scripts (copy/paste)

Create a folder like `C:\ec2-scheduler\` and put these files there.

#### `start-ec2.ps1`

```powershell
param(
  [string]$Region = "ap-south-1",
  [string]$InstanceId = "i-xxxxxxxxxxxxxxxxx",
  [string]$Profile = "ec2-scheduler",
  [string]$LogFile = "C:\ec2-scheduler\ec2-scheduler.log"
)

$ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
Add-Content -Path $LogFile -Value "[$ts] START requested for $InstanceId ($Region)"

aws --profile $Profile --region $Region ec2 start-instances --instance-ids $InstanceId | Out-Null
aws --profile $Profile --region $Region ec2 wait instance-running --instance-ids $InstanceId

$ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
Add-Content -Path $LogFile -Value "[$ts] START completed for $InstanceId"
```

#### `stop-ec2.ps1`

```powershell
param(
  [string]$Region = "ap-south-1",
  [string]$InstanceId = "i-xxxxxxxxxxxxxxxxx",
  [string]$Profile = "ec2-scheduler",
  [string]$LogFile = "C:\ec2-scheduler\ec2-scheduler.log"
)

$ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
Add-Content -Path $LogFile -Value "[$ts] STOP requested for $InstanceId ($Region)"

aws --profile $Profile --region $Region ec2 stop-instances --instance-ids $InstanceId | Out-Null
aws --profile $Profile --region $Region ec2 wait instance-stopped --instance-ids $InstanceId

$ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
Add-Content -Path $LogFile -Value "[$ts] STOP completed for $InstanceId"
```

### Create scheduled tasks (Task Scheduler)

Create **two tasks**: one Start and one Stop.

- **General**
  - Name: `EC2 Start` / `EC2 Stop`
  - “Run whether user is logged on or not” (recommended)
  - “Run with highest privileges” (optional)
- **Triggers**
  - Set your desired time(s) and days
- **Actions**
  - Program/script: `powershell.exe`
  - Add arguments (Start):
    - `-NoProfile -ExecutionPolicy Bypass -File "C:\ec2-scheduler\start-ec2.ps1" -Region ap-south-1 -InstanceId i-xxxxxxxxxxxxxxxxx -Profile ec2-scheduler`
  - Add arguments (Stop):
    - `-NoProfile -ExecutionPolicy Bypass -File "C:\ec2-scheduler\stop-ec2.ps1" -Region ap-south-1 -InstanceId i-xxxxxxxxxxxxxxxxx -Profile ec2-scheduler`
  - Start in: `C:\ec2-scheduler\`

### Notes / gotchas (Windows)

- Task Scheduler uses your **Windows local timezone**.
- If your PC is asleep/off, tasks won’t run unless you configure wake settings.
- If you need the EC2 to start your trading bot reliably after instance boot, consider:
  - starting the instance a few minutes earlier than the bot’s cron time, or
  - adding an `@reboot` job / `systemd` unit on the instance (see `docs/cron_usage.md` for current in-instance scheduling patterns).

## Quick recommendation for this repo

- Use **EventBridge Scheduler + Lambda** for **instance start/stop**.
- Keep using `docs/cron_usage.md` for **in-instance bot scheduling**.
- Start the instance **before** your bot’s in-instance cron time so the cron tick isn’t missed during boot.


