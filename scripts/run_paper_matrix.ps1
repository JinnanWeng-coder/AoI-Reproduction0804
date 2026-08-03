param(
    [switch]$DryRun,
    [switch]$Execute,
    [string]$Device = "auto"
)
$root = Split-Path -Parent $PSScriptRoot
if ($DryRun -eq $Execute) { throw "Choose exactly one of -DryRun or -Execute" }
if ($DryRun) {
    & python (Join-Path $root "scripts\matrix_runner.py") --profile paper_faithful --dry-run
    exit $LASTEXITCODE
}
& python (Join-Path $root "scripts\matrix_runner.py") --profile paper_faithful --execute --device $Device
exit $LASTEXITCODE

