$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$Jar = Join-Path $ProjectDir "target\gomoku-ml-training-1.0.0.jar"
$ConfigDir = if ($env:GOMOKU_CONFIG_DIR) {
    $env:GOMOKU_CONFIG_DIR
} else {
    Join-Path $env:USERPROFILE ".config\gomoku"
}
$ConfigFile = Join-Path $ConfigDir "client.conf"

function Read-ServerUrl {
    if (-not (Test-Path -LiteralPath $ConfigFile)) {
        return ""
    }
    $line = Get-Content -LiteralPath $ConfigFile | Where-Object {
        $_ -like "SERVER_URL=*"
    } | Select-Object -Last 1
    if ($line) {
        return $line.Substring("SERVER_URL=".Length)
    }
    return ""
}

function Save-ServerUrl($url) {
    New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
    Set-Content -LiteralPath $ConfigFile -Value "SERVER_URL=$url" -Encoding UTF8
}

function Ensure-Jar {
    if (-not (Test-Path -LiteralPath $Jar)) {
        Push-Location $ProjectDir
        try {
            mvn -q -DskipTests package
        } finally {
            Pop-Location
        }
    }
}

if ($args.Count -eq 0 -or $args[0] -eq "download") {
    $serverUrl = if ($env:GOMOKU_SERVER_URL) {
        $env:GOMOKU_SERVER_URL
    } else {
        Read-ServerUrl
    }
    if ([string]::IsNullOrWhiteSpace($serverUrl)) {
        $serverUrl = Read-Host "Server URL (for example http://SERVER_IP:5000)"
        if ([string]::IsNullOrWhiteSpace($serverUrl)) {
            throw "Server URL is required."
        }
        Save-ServerUrl $serverUrl
    }
    $outputDir = if ($env:GOMOKU_DOWNLOAD_DIR) {
        $env:GOMOKU_DOWNLOAD_DIR
    } else {
        Join-Path (Get-Location) "gomoku-downloads"
    }
    $password = Read-Host "Password" -AsSecureString
    $credential = New-Object System.Management.Automation.PSCredential("gomoku", $password)
    $plainPassword = $credential.GetNetworkCredential().Password
    Ensure-Jar
    $env:GOMOKU_DATA_TOKEN = $plainPassword
    & java -jar $Jar download-data --url $serverUrl --output $outputDir
    Remove-Item Env:GOMOKU_DATA_TOKEN -ErrorAction SilentlyContinue
    exit $LASTEXITCODE
}

if ($args[0] -eq "help" -or $args[0] -eq "--help" -or $args[0] -eq "-h") {
    Write-Host "Usage: gomoku download"
    Write-Host "       gomoku config"
    exit 0
}

if ($args[0] -eq "config") {
    $serverUrl = Read-Host "Server URL (for example http://SERVER_IP:5000)"
    if ([string]::IsNullOrWhiteSpace($serverUrl)) {
        throw "Server URL is required."
    }
    Save-ServerUrl $serverUrl
    Write-Host "Saved server URL to $ConfigFile"
    exit 0
}

Write-Host "Usage: gomoku download"
exit 2
