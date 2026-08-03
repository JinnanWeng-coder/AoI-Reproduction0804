param(
    [string]$Device = "cpu",
    [string]$RunName = "smoke_paper_faithful_p05_n04_g25_seed02",
    [string]$Python = "python"
)
$root = Split-Path -Parent $PSScriptRoot
& $Python (Join-Path $root "Main.py") --profile paper_faithful --scenario p05_n04_g25 --seed 2 --device $Device --smoke --run-name $RunName
exit $LASTEXITCODE
