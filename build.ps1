param(
    [string]$Abi = "arm64-v8a",
    [string]$BuildType = "Release",
    [string]$Ndk = "",
    [Parameter(Mandatory = $true)]
    [string]$PreloaderRoot,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$SourceDir = $PSScriptRoot

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Resolve-NdkPath {
    param([string]$ExplicitNdk)

    $Candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($ExplicitNdk)) { $Candidates += $ExplicitNdk }
    if (-not [string]::IsNullOrWhiteSpace($env:ANDROID_NDK_HOME)) { $Candidates += $env:ANDROID_NDK_HOME }
    if (-not [string]::IsNullOrWhiteSpace($env:ANDROID_NDK_ROOT)) { $Candidates += $env:ANDROID_NDK_ROOT }

    $SdkRoots = @($env:ANDROID_HOME, $env:ANDROID_SDK_ROOT) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique
    foreach ($SdkRoot in $SdkRoots) {
        $NdkRoot = Join-Path $SdkRoot "ndk"
        if (Test-Path $NdkRoot) {
            Get-ChildItem -LiteralPath $NdkRoot -Directory | Sort-Object Name -Descending |
                ForEach-Object { $Candidates += $_.FullName }
        }
        $NdkBundle = Join-Path $SdkRoot "ndk-bundle"
        if (Test-Path $NdkBundle) { $Candidates += $NdkBundle }
    }

    foreach ($Candidate in ($Candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        $Toolchain = Join-Path $Candidate "build\cmake\android.toolchain.cmake"
        if (Test-Path $Toolchain) { return Get-FullPath $Candidate }
    }
    throw "Android NDK not found. Pass -Ndk or set ANDROID_NDK_HOME/ANDROID_NDK_ROOT/ANDROID_HOME/ANDROID_SDK_ROOT."
}

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string]$FilePath,
          [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$FilePath failed with exit code $LASTEXITCODE" }
}

$PreloaderRoot = Get-FullPath $PreloaderRoot
if (-not (Test-Path (Join-Path $PreloaderRoot "include\pl\Mod.hpp"))) {
    throw "PreloaderRoot does not look like a preloader-android checkout: $PreloaderRoot"
}

$Ndk = Resolve-NdkPath $Ndk
$Toolchain = Join-Path $Ndk "build\cmake\android.toolchain.cmake"

$BuildDir = Join-Path $SourceDir "build\android-$Abi-$BuildType"
if ($Clean -and (Test-Path $BuildDir)) { Remove-Item -LiteralPath $BuildDir -Recurse -Force }

Invoke-Checked cmake `
    -S $SourceDir -B $BuildDir -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DANDROID_ABI=$Abi" `
    "-DANDROID_PLATFORM=android-24" `
    "-DCMAKE_BUILD_TYPE=$BuildType" `
    "-DPRELOADER_ANDROID_ROOT=$PreloaderRoot"

Invoke-Checked cmake --build $BuildDir --target framegen_mod

$OutDir = Join-Path $BuildDir "out\$Abi"
$DistDir = Join-Path $SourceDir "dist\$Abi"
if (Test-Path $DistDir) { Remove-Item -LiteralPath $DistDir -Recurse -Force }
$PackageDir = Join-Path $DistDir "framegen-mod"
New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null

Copy-Item -LiteralPath (Join-Path $OutDir "libframegen_mod.so") -Destination $PackageDir
Copy-Item -LiteralPath (Join-Path $SourceDir "manifest.json") -Destination $PackageDir

$ArchivePath = Join-Path $DistDir "framegen-mod.levipack"
$TempZipPath = Join-Path $DistDir "framegen-mod.zip"
Compress-Archive -Path (Join-Path $PackageDir "*") -DestinationPath $TempZipPath -Force
Move-Item -LiteralPath $TempZipPath -Destination $ArchivePath -Force

Write-Host "Built:"
Write-Host "  $PackageDir"
Write-Host "  $ArchivePath"
