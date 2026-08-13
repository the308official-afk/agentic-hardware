# AWS Sync Scripts

These scripts sync this repo to and from the EC2 machine used for the direct SGLang KV instrumentation testbed.

## Configure

Edit:

```text
aws/config.sh
```

or export values before running:

```bash
export AGENTIC_HW_SERVERS="1.2.3.4"
export AGENTIC_HW_PEM="/Users/oluwolejaiyeoba/Documents/GitHub/secrets/projectonekeypair.pem"
export AGENTIC_HW_EC2_USER="ec2-user"
```

The remote project path defaults to:

```text
/home/ec2-user/agentic_hardware
```

Override it with:

```bash
export AGENTIC_HW_REMOTE_DIR="/home/ec2-user/agentic_hardware"
```

## Upload Code

```bash
./aws/upload.sh 0
```

This uploads the repo while excluding generated files, virtual environments, and artifacts.

## SSH

```bash
./aws/ssh_to_ec2.sh 0
```

## Check EC2 Readiness

```bash
./aws/check_ec2_ready.sh 0
```

This checks SSH, disk, Python, `nvidia-smi`, and PyTorch CUDA visibility.

## Download Results

```bash
./aws/download.sh 0
```

This downloads:

```text
remote: ~/agentic_hardware/sglang_direct_kv/artifacts/
local:  ./sglang_direct_kv/artifacts/
```

## Typical Flow

From local machine:

```bash
export AGENTIC_HW_SERVERS="<ec2-public-ip>"
./aws/upload.sh 0
./aws/ssh_to_ec2.sh 0
```

On EC2:

```bash
cd ~/agentic_hardware/sglang_direct_kv
bash scripts/setup_ec2.sh
source .venv/bin/activate
python scripts/probe_sglang_kv_paths.py --out artifacts/sglang_probe.json
```

Back on local machine:

```bash
./aws/download.sh 0
```
