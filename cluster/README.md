# Slurm workflow

These portable scripts keep every workload off the login node. Submit from a
fresh, versioned project directory in your own cluster scratch space. Supply
site-specific account, partition, and QoS options to `sbatch` at submission
time; do not commit personal cluster routing or credentials.

Run `setup_cpu.sbatch` first. It creates an isolated virtual environment, runs
all tests, and records the random and scripted controls. Submit only one job at
a time and inspect its log before continuing.

Run `check_gpu.sbatch` before training when the environment or driver changes.
It requests one GPU for five minutes and verifies driver visibility, PyTorch
CUDA support, device identity, and an actual tensor operation.

`train_v100.sbatch` requests one V100 for at most two hours. It creates a
job-ID-specific run directory, generates disjoint training and evaluation
leagues, trains one 500,000-step Maskable PPO seed with eight subprocess
environments, and evaluates 500 unseen episodes. A rerun cannot overwrite an
earlier run directory.

The initial V100 experiment is a utilization measurement as well as a learning
test. This small MLP may remain CPU-bound. Do not schedule a larger sweep until
the run shows a learning signal and the logs show that GPU use is justified.

## Older GPUs

Recent PyTorch CUDA builds may stop compiling kernels for older accelerator
architectures even when the driver can see the device. The versioned
`cuda126_v1` workflow creates a separate environment using PyTorch's CUDA 12.6
wheel, asserts that `sm_70` kernels are present, and only then tests or trains on
a V100. It never modifies the default environment.

## Paired continuation experiment

`round2.sbatch` uses an existing personal environment and a saved first-run
model. It runs the two objectives sequentially for three continuation seeds,
with independent validation and final test pools. It defaults to CPU; supply
`SAP_RL_DEVICE=cuda` and request a GPU explicitly to test GPU execution. See
[the full protocol and completed local results](../docs/ROUND2.md).
