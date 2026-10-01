[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InputJson,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$VoiceTokenId = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0',
    [ValidateRange(-10, 10)][int]$Rate = 0,
    [ValidateRange(0, 100)][int]$Volume = 100
)

# Offline, plain-text SAPI synthesis. Input is a JSON array containing only
# id, text, variant (original|normalized), and pair_id. No normalization or
# evaluation labels are read here. Existing output directories are rejected.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Get-TextSha256([string]$Value) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($utf8.GetBytes($Value)))).Replace('-', '').ToLowerInvariant()
    } finally { $sha.Dispose() }
}

function Write-RunJson([string]$Path, $Value) {
    [System.IO.File]::WriteAllText($Path, (($Value | ConvertTo-Json -Depth 16) + "`n"), $utf8)
}

function Get-WaveEvidence([string]$Path) {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 44 -or [Text.Encoding]::ASCII.GetString($bytes, 0, 4) -ne 'RIFF' -or
        [Text.Encoding]::ASCII.GetString($bytes, 8, 4) -ne 'WAVE') { throw 'Invalid RIFF/WAVE output.' }
    if ([BitConverter]::ToUInt32($bytes, 4) + 8 -ne $bytes.Length) { throw 'RIFF size does not match file length.' }
    $offset = 12
    $format = $null
    $dataOffset = -1
    $dataLength = 0
    while ($offset + 8 -le $bytes.Length) {
        $name = [Text.Encoding]::ASCII.GetString($bytes, $offset, 4)
        $length = [long][BitConverter]::ToUInt32($bytes, $offset + 4)
        $start = $offset + 8
        if ($start + $length -gt $bytes.Length) { throw 'WAVE chunk exceeds file length.' }
        if ($name -eq 'fmt ') {
            if ($null -ne $format -or $length -lt 16) { throw 'Invalid or duplicate WAVE format chunk.' }
            $format = [ordered]@{
                format_tag = [int][BitConverter]::ToUInt16($bytes, $start)
                channels = [int][BitConverter]::ToUInt16($bytes, $start + 2)
                sample_rate_hz = [int][BitConverter]::ToUInt32($bytes, $start + 4)
                byte_rate = [int][BitConverter]::ToUInt32($bytes, $start + 8)
                block_align = [int][BitConverter]::ToUInt16($bytes, $start + 12)
                bits_per_sample = [int][BitConverter]::ToUInt16($bytes, $start + 14)
            }
        } elseif ($name -eq 'data') {
            if ($dataOffset -ge 0) { throw 'Unexpected multiple WAVE data chunks.' }
            $dataOffset = $start
            $dataLength = $length
        }
        $offset = $start + $length + ($length % 2)
    }
    if ($null -eq $format -or $format.format_tag -ne 1 -or $format.channels -ne 1 -or
        $format.sample_rate_hz -ne 22050 -or $format.bits_per_sample -ne 16 -or
        $format.block_align -ne 2 -or $format.byte_rate -ne 44100) {
        throw 'Output must be PCM signed 16-bit, 22050 Hz, mono.'
    }
    if ($dataOffset -lt 0 -or $dataLength -le 0 -or $dataLength % 2 -ne 0) { throw 'Empty or incomplete PCM output.' }
    $nonzero = 0
    $peak = 0
    for ($index = $dataOffset; $index -lt $dataOffset + $dataLength; $index += 2) {
        $sample = [int][BitConverter]::ToInt16($bytes, $index)
        if ($sample -ne 0) { $nonzero++ }
        $peak = [Math]::Max($peak, [Math]::Abs($sample))
    }
    if ($nonzero -eq 0) { throw 'PCM output is entirely zero; not a successful synthesis.' }
    return [ordered]@{
        format = $format
        file_bytes = $bytes.Length
        pcm_bytes = $dataLength
        pcm_samples = $dataLength / 2
        duration_seconds = $dataLength / [double]$format.byte_rate
        nonzero_pcm_samples = $nonzero
        peak_absolute_pcm16 = $peak
        wav_sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$destination = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $destination) { throw "Output directory already exists; refusing overwrite: $destination" }
$null = New-Item -ItemType Directory -Path $destination
$runPath = Join-Path $destination 'run.json'
$state = [ordered]@{
    schema_version = 1
    experiment = 'reading-demo-sapi-v1'
    status = 'started'
    phase = 'preflight'
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    backend = 'Windows SAPI'
    offline = $true
    external_api_cost_usd = 0
    training = $false
    playback_performed = $false
    listening_review = 'not_performed'
    pronunciation_pass_claimed = $false
    rate = $Rate
    volume = $Volume
    speak_flags = 16
    speak_flags_name = 'SVSFIsNotXML (synchronous plain text)'
    sapi_audio_format_type = 22
    requested_format = 'PCM signed 16-bit little-endian, 22050 Hz, mono'
    voice_token_requested = $VoiceTokenId
    os_version = [Environment]::OSVersion.VersionString
    powershell_version = $PSVersionTable.PSVersion.ToString()
    clr_version = [Environment]::Version.ToString()
    process_is_64bit = [Environment]::Is64BitProcess
    script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
Write-RunJson $runPath $state
$voice = $null
$writer = $null
$records = @()
try {
    $inputPath = (Resolve-Path -LiteralPath $InputJson).ProviderPath
    $inputText = [System.IO.File]::ReadAllText($inputPath, $utf8)
    $items = @($inputText | ConvertFrom-Json)
    if (-not $inputText.TrimStart().StartsWith('[') -or $items.Count -lt 1 -or $items.Count -gt 100) {
        throw 'Input must be a JSON array with 1 to 100 items.'
    }
    $seen = @{}
    foreach ($item in $items) {
        $keys = @($item.PSObject.Properties.Name | Sort-Object)
        if (($keys -join ',') -ne 'id,pair_id,text,variant') { throw 'Each input item must contain exactly id, pair_id, text, variant.' }
        if ($item.id -isnot [string] -or $item.id -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$' -or $seen.ContainsKey($item.id)) {
            throw 'Input ids must be unique safe ASCII identifiers.'
        }
        if ($item.pair_id -isnot [string] -or [string]::IsNullOrWhiteSpace($item.pair_id) -or
            $item.variant -isnot [string] -or $item.variant -notin @('original', 'normalized') -or
            $item.text -isnot [string] -or [string]::IsNullOrWhiteSpace($item.text)) { throw 'Invalid pair_id, variant, or text.' }
        $seen[$item.id] = $true
    }
    $state.input_path = $inputPath
    $state.input_sha256 = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $state.input_items = $items.Count
    $voice = New-Object -ComObject SAPI.SpVoice
    $tokens = @($voice.GetVoices())
    $selected = @($tokens | Where-Object { $_.Id -eq $VoiceTokenId })
    if ($selected.Count -ne 1) { throw 'Requested installed SAPI voice token is unavailable.' }
    $voice.Voice = $selected[0]
    $voice.Rate = $Rate
    $voice.Volume = $Volume
    $state.voice = [ordered]@{
        token_id = $voice.Voice.Id
        name = $voice.Voice.GetDescription()
        language_lcid_hex = $voice.Voice.GetAttribute('Language')
        gender = $voice.Voice.GetAttribute('Gender')
        vendor = $voice.Voice.GetAttribute('Vendor')
    }
    $sapiDll = Join-Path ([Environment]::SystemDirectory) 'Speech\Common\sapi.dll'
    if (Test-Path -LiteralPath $sapiDll) {
        $state.sapi_binary = [ordered]@{path=$sapiDll;version=(Get-Item -LiteralPath $sapiDll).VersionInfo.FileVersion;sha256=(Get-FileHash -LiteralPath $sapiDll -Algorithm SHA256).Hash.ToLowerInvariant()}
    }
    $state.phase = 'synthesis'
    Write-RunJson $runPath $state
    $jsonlPath = Join-Path $destination 'outputs.jsonl'
    $writer = New-Object System.IO.StreamWriter([System.IO.File]::Open($jsonlPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write), $utf8)
    $number = 0
    foreach ($item in $items) {
        $number++
        $wavPath = Join-Path $destination ('{0:D3}-{1}-{2}.wav' -f $number, $item.id, $item.variant)
        $record = [ordered]@{id=$item.id;pair_id=$item.pair_id;variant=$item.variant;text=$item.text;text_sha256=(Get-TextSha256 $item.text);text_utf16_code_units=$item.text.Length;status='started';wav_path=$wavPath}
        $stream = $null
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            if (Test-Path -LiteralPath $wavPath) { throw 'WAVE file already exists; refusing overwrite.' }
            $stream = New-Object -ComObject SAPI.SpFileStream
            $stream.Format.Type = 22
            $stream.Open($wavPath, 3, $false)
            $voice.AudioOutputStream = $stream
            # Flag 16 explicitly disables XML interpretation; no SSML or auto-detect.
            $null = $voice.Speak($item.text, 16)
            $stream.Close()
            $record.evidence = Get-WaveEvidence $wavPath
            $record.status = 'generated'
        } catch {
            $record.status = 'failed'
            $record.error_type = $_.Exception.GetType().FullName
            $record.error = $_.Exception.Message
        } finally {
            $timer.Stop()
            $record.elapsed_ms = $timer.Elapsed.TotalMilliseconds
            if ($null -ne $stream) {
                try { $voice.AudioOutputStream = $null } catch { }
                try { $stream.Close() } catch { }
                $null = [Runtime.InteropServices.Marshal]::FinalReleaseComObject($stream)
            }
        }
        $records += $record
        $writer.WriteLine(($record | ConvertTo-Json -Depth 12 -Compress))
        $writer.Flush()
        $state.completed_items = $records.Count
        Write-RunJson $runPath $state
    }
    $writer.Dispose()
    $writer = $null
    $state.generated_items = @($records | Where-Object { $_.status -eq 'generated' }).Count
    $state.failed_items = @($records | Where-Object { $_.status -ne 'generated' }).Count
    $state.outputs_sha256 = (Get-FileHash -LiteralPath $jsonlPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $state.input_sha256 -or
        (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $state.script_sha256) {
        throw 'Input or synthesis script changed during execution.'
    }
    if ($state.failed_items -gt 0) { throw 'One or more samples failed; per-item evidence retained in outputs.jsonl.' }
    $state.phase = 'complete'
    $state.status = 'complete'
} catch {
    $state.status = 'failed'
    $state.error_type = $_.Exception.GetType().FullName
    $state.error = $_.Exception.Message
    throw
} finally {
    if ($null -ne $writer) { $writer.Dispose() }
    if ($null -ne $voice) { $null = [Runtime.InteropServices.Marshal]::FinalReleaseComObject($voice) }
    $state.finished_at_utc = [DateTime]::UtcNow.ToString('o')
    Write-RunJson $runPath $state
}
[ordered]@{output_directory=$destination;status=$state.status;generated_items=$state.generated_items;failed_items=$state.failed_items;listening_review=$state.listening_review} | ConvertTo-Json -Compress
