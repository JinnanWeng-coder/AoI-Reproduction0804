param(
    [switch]$DryRun,
    [switch]$Execute,
    [string]$Device = "auto",
    [ValidateRange(1, 2147483647)][int]$CheckpointEvery = 5,
    [switch]$RecoverEmptyRun,
    [ValidateSet("train", "eval", "audit", "all")][string]$Stage = "train",
    [ValidateSet("validation", "final_test")][string]$EvalPurpose = "validation",
    [int]$EvalEpisodes = 100,
    [string]$EvalSeeds = "",
    [string]$OutputRoot = "experiments/runs",
    [string]$LogDir = "batch_logs",
    [string]$Report = "",
    [string]$Python = "python"
)
$root = Split-Path -Parent $PSScriptRoot
if ($DryRun -eq $Execute) { throw "Choose exactly one of -DryRun or -Execute" }
$arguments = @("--profile", "paper_faithful", "--device", $Device, "--checkpoint-every", $CheckpointEvery, "--stage", $Stage, "--eval-purpose", $EvalPurpose, "--eval-episodes", $EvalEpisodes, "--output-root", $OutputRoot, "--log-dir", $LogDir)
if ($RecoverEmptyRun) { $arguments += "--recover-empty-run" }
if ($EvalSeeds -ne "") { $arguments += @("--eval-seeds", $EvalSeeds) }
if ($Report -ne "") { $arguments += @("--report", $Report) }
if ($DryRun) {
    & $Python (Join-Path $root "scripts\matrix_runner.py") @arguments --dry-run
    exit $LASTEXITCODE
}
& $Python (Join-Path $root "scripts\matrix_runner.py") @arguments --execute
exit $LASTEXITCODE
