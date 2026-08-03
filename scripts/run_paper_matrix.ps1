param(
    [switch]$DryRun,
    [switch]$Execute,
    [string]$Device = "auto",
    [string]$Python = "python"
)
$root = Split-Path -Parent $PSScriptRoot
if ($DryRun -eq $Execute) { throw "Choose exactly one of -DryRun or -Execute" }
if ($DryRun) {
    & $Python (Join-Path $root "scripts\matrix_runner.py") --profile paper_faithful --dry-run
    exit $LASTEXITCODE
}
& $Python (Join-Path $root "scripts\matrix_runner.py") --profile paper_faithful --execute --device $Device
exit $LASTEXITCODE
