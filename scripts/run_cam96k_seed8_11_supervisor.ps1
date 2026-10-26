param(
    [Parameter(Mandatory=$true)][ValidateSet('train','evaluate')][string]$Phase,
    [Parameter(Mandatory=$true)][string]$BatchRoot,
    [Parameter(Mandatory=$true)][string]$ExpectedCommit,
    [Parameter(Mandatory=$true)][string]$Receipt,
    [string]$Python = 'E:\Jinnan_Edisk\Research\Parvini_Reproduction_core\.envs\aoi_cuda310\python.exe',
    [ValidateRange(1,8)][int]$EvalWorkers = 8,
    [switch]$ResumeIncomplete,
    [string]$Reason
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$started = (Get-Date).ToString('o')
$exitCode = 1
$validationExitCode = $null
$errorText = $null
try {
    Set-Location -LiteralPath $repo
    if (Test-Path -LiteralPath $Receipt) { throw 'Receipt already exists; use a new launch identity' }
    $receiptParent = Split-Path -Parent $Receipt
    if (-not (Test-Path -LiteralPath $receiptParent -PathType Container)) { throw 'Receipt parent directory is missing' }
    $head = (git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $head -ne $ExpectedCommit) { throw 'HEAD mismatch' }
    $manifest = Get-Content -LiteralPath (Join-Path $BatchRoot 'protocol\manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schema -ne 'cam96k_seed8_11_full_state_v1' -or $manifest.implementation_commit -ne $ExpectedCommit) { throw 'Wrong schema or pinned batch commit' }
    if (-not $manifest.smoke) {
        $dirty = git status --porcelain --untracked-files=all
        if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'Formal execution requires a clean checkout' }
    }
    if ($Phase -eq 'evaluate') {
        & $Python -B -m analysis.cam96k_seed8_11 verify-training --batch-root $BatchRoot
        if ($LASTEXITCODE -ne 0) { $exitCode = $LASTEXITCODE; throw 'Twenty-run training gate failed' }
    }
    $arguments = @('-B','-u','-m','analysis.cam96k_seed8_11',$Phase,'--batch-root',$BatchRoot)
    if ($Phase -eq 'evaluate') { $arguments += @('--eval-workers', [string]$EvalWorkers) }
    if ($ResumeIncomplete) {
        if ($Phase -ne 'train' -or -not $Reason) { throw 'ResumeIncomplete requires train phase and a technical Reason' }
        $arguments += @('--resume-incomplete','--reason',$Reason)
    }
    & $Python @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) { throw ('Python supervisor failed: ' + $exitCode) }
    $validation = if ($Phase -eq 'train') { 'verify-training' } else { 'summary' }
    & $Python -B -m analysis.cam96k_seed8_11 $validation --batch-root $BatchRoot
    $validationExitCode = $LASTEXITCODE
    if ($validationExitCode -ne 0) { $exitCode = $validationExitCode; throw ('Final validation failed: ' + $validationExitCode) }
} catch {
    $errorText = $_.Exception.Message
    if ($exitCode -eq 0) { $exitCode = 1 }
    [Console]::Error.WriteLine($errorText)
} finally {
    $data = [ordered]@{phase=$Phase; pid=$PID; batch_root=$BatchRoot; expected_commit=$ExpectedCommit;
        started_at=$started; finished_at=(Get-Date).ToString('o'); exit_code=$exitCode;
        validation_exit_code=$validationExitCode; error=$errorText; eval_workers=$EvalWorkers}
    # The launch identity is unique. Publish a receipt only after the process and
    # post-check finish; absence of this receipt is never interpreted as success.
    if (-not (Test-Path -LiteralPath $Receipt)) {
        $temporary = $Receipt + '.tmp'
        $data | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
        Move-Item -LiteralPath $temporary -Destination $Receipt -ErrorAction Stop
    }
}
exit $exitCode
