[CmdletBinding()]
param(
    [switch]$Execute,
    [string]$Prepared,
    [string]$RunRoot = 'E:\CodexRuntime\ReadCue\runs\model-experiment-v2-2026-10-27',
    [string]$RuntimeRoot = 'E:\CodexRuntime\ReadCue',
    [string]$Distribution = 'Ubuntu'
)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
if (-not $Prepared) { $Prepared = Join-Path $project 'runs\model-experiment-v2-prepared-2026-10-27' }

function Convert-ToWslPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if ($absolute -notmatch '^([A-Za-z]):[\\/](.*)$') { throw "A Windows drive path is required: $Path" }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\', '/')
}

$p = Convert-ToWslPath $project
$preparedPath = Convert-ToWslPath $Prepared
$runPath = Convert-ToWslPath $RunRoot
$runtimePath = Convert-ToWslPath $RuntimeRoot
$oldTrain = "$runtimePath/runs/model-experiment-v1-2026-10-01/train"
$inputs = "$preparedPath/heldout/inputs.jsonl"
$proof = "$preparedPath/heldout/freeze-proof.json"
$task = "$p/configs/model_task_v1.json"
$common = @('--project', $p, '--model-dir', "$runtimePath/models/qwen3-0.6b", '--task-config', $task, '--budget-root', $runPath)
$prefix = @('--distribution', $Distribution, '--cd', $p, '--exec', 'env', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONIOENCODING=utf-8', "PYTHONPATH=$p/src", 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'TOKENIZERS_PARALLELISM=false', "$runtimePath/.venv-linux/bin/python", '-B')
$steps = @(
    @{name='preflight'; argv=@("$p/scripts/run_model_experiment.py", 'preflight') + $common + @('--dataset', "$preparedPath/dataset", '--output', "$runPath/preflight")},
    @{name='train'; argv=@("$p/scripts/run_model_experiment.py", 'train') + $common + @('--dataset', "$preparedPath/dataset", '--heldout-proof', $proof, '--output', "$runPath/train")},
    @{name='base'; argv=@("$p/scripts/run_model_experiment.py", 'predict') + $common + @('--inputs', $inputs, '--output', "$runPath/base")},
    @{name='previous'; argv=@("$p/scripts/run_model_experiment.py", 'predict') + $common + @('--inputs', $inputs, '--adapter', $oldTrain, '--output', "$runPath/previous")},
    @{name='adapted'; argv=@("$p/scripts/run_model_experiment.py", 'predict') + $common + @('--inputs', $inputs, '--adapter', "$runPath/train", '--output', "$runPath/adapted")},
    @{name='rules'; argv=@("$p/scripts/run_audited_rules.py", '--inputs', $inputs, '--resources', "$runtimePath/data/comment-policy-resources-2026-10-01/resources.json", '--semantic-resources', "$runtimePath/data/semantic-emoji-resources-v1-2026-10-01/resources.json", '--lexicon', "$runtimePath/data/reading-lexicon-v2-2026-10-01/lexicon.json", '--context-resources', "$runtimePath/data/context-reading-resources-v1-2026-10-01/resources.json", '--output', "$runPath/rules")},
    @{name='evaluation'; argv=@("$p/tools/score_model_experiment.py", '--project', $p, '--training-run', "$runPath/train", '--heldout', "$preparedPath/heldout", '--heldout-proof', $proof, '--base', "$runPath/base/predictions.jsonl", '--adapted', "$runPath/adapted/predictions.jsonl", '--previous', "$runPath/previous/predictions.jsonl", '--previous-training-run', $oldTrain, '--rules', "$runPath/rules/predictions.jsonl", '--output', "$runPath/evaluation")}
)
if (-not $Execute) {
    $steps | ForEach-Object { [pscustomobject]@{stage=$_.name; executable='wsl.exe'; arguments=@($prefix + $_.argv)} } | ConvertTo-Json -Depth 5
    return
}
# The model runner also validates the output root, reparse points and budget.
$resolvedRoot = [IO.Path]::GetFullPath($RunRoot).TrimEnd('\')
$resolvedProject = [IO.Path]::GetFullPath($project).TrimEnd('\')
if ($resolvedRoot -eq $resolvedProject -or $resolvedRoot.StartsWith($resolvedProject + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'RunRoot must be outside the Git checkout.' }
foreach ($name in @('dataset\manifest.json', 'heldout\freeze-proof.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $Prepared $name) -PathType Leaf)) { throw "Prepared evidence is missing: $name" }
}
if (Test-Path -LiteralPath $RunRoot) { throw 'RunRoot must be new; existing runs and failure receipts are preserved.' }
New-Item -ItemType Directory -Path $RunRoot | Out-Null
foreach ($step in $steps) {
    $arguments = @($prefix + $step.argv)
    $receipt = [ordered]@{stage=$step.name; executable='wsl.exe'; arguments=$arguments; status='started'; started_at_utc=[DateTime]::UtcNow.ToString('o')}
    $receiptPath = Join-Path $RunRoot ($step.name + '-launch.json')
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
    & wsl.exe @arguments
    $code = $LASTEXITCODE
    $receipt.status = if ($code -eq 0) { 'complete' } else { 'failed' }
    $receipt.returncode = $code
    $receipt.finished_at_utc = [DateTime]::UtcNow.ToString('o')
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
    if ($code -ne 0) { throw "Stage $($step.name) failed with exit code $code. Evidence retained in $RunRoot" }
}
Write-Output 'Training and automatic scoring completed. Separate anonymous semantic review is still required.'
