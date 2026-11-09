# ============================================================
#  nlpbase 环境一键恢复脚本（Windows / PowerShell）
#
#  用法：
#      .\setup.ps1              # 有 NVIDIA 显卡，装 CUDA 版
#      .\setup.ps1 -Index cu128 # 驱动较老时换 CUDA 版本
#      .\setup.ps1 -Cpu         # 没有 NVIDIA 显卡，装 CPU 版
#      .\setup.ps1 -DryRun      # 只看会做什么，不真的安装
#
#  只做四件事：建 conda 环境 → 装 torch → 装其余依赖 → 跑验证。
#  任何一步失败都会停下并打印原因。
# ============================================================
param(
    [string]$EnvName = "nlpbase",
    [string]$PythonVer = "3.10",
    [string]$Index = "cu130",
    [switch]$Cpu,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8

function Info($msg) { Write-Host "[*] $msg" -ForegroundColor Cyan }
function Good($msg) { Write-Host "[ok] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Plan($msg) { Write-Host "    (dry-run) $msg" -ForegroundColor DarkGray }
function Fail($msg) {
    Write-Host "[x] $msg" -ForegroundColor Red
    exit 1
}

# ---------- 1. 检查 conda ----------
Info "检查 conda"
$condaBin = $null
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if ($null -ne $condaCmd) {
    $condaBin = $condaCmd.Source
} else {
    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "D:\miniconda3\Scripts\conda.exe",
        "D:\Anaconda3\Scripts\conda.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { $condaBin = $c; break }
    }
}
if ($null -eq $condaBin) {
    Fail "找不到 conda。请先安装 Miniconda，或改用 Anaconda Prompt 运行本脚本。"
}
Good ("conda: " + $condaBin)

# ---------- 2. 判断显卡 ----------
$hasNvidia = $false
if ($Cpu) {
    Info "按参数要求使用 CPU 版"
} else {
    Info "检查 NVIDIA 显卡"
    $smiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -ne $smiCmd) {
        $gpuText = nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
        if ($gpuText) {
            $hasNvidia = $true
            Good ("显卡: " + ($gpuText -join " / "))
        }
    }
    if (-not $hasNvidia) {
        Warn "未检测到 NVIDIA 显卡，改为安装 CPU 版"
    }
}

# ---------- 3. 建环境 ----------
Info ("conda 环境 " + $EnvName + " (Python " + $PythonVer + ")")
$envList = & $condaBin env list
$exists = $envList -match ("^\s*" + [regex]::Escape($EnvName) + "\s")
if ($exists) {
    Warn ("环境 " + $EnvName + " 已存在，跳过创建")
    Warn ("要重建请先执行: conda env remove -n " + $EnvName)
} else {
    $createArgs = @("create", "-y", "-n", $EnvName, ("python=" + $PythonVer))
    if ($DryRun) {
        Plan ("conda " + ($createArgs -join " "))
    } else {
        & $condaBin @createArgs
        if ($LASTEXITCODE -ne 0) {
            Warn "创建环境失败，可能是 .condarc 里的镜像地址失效"
            Warn "绕过镜像配置重试（直接用清华源）"
            & $condaBin clean -i -y | Out-Null
            $fallback = @(
                "create", "-y", "-n", $EnvName, ("python=" + $PythonVer),
                "--override-channels",
                "-c", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
            )
            & $condaBin @fallback
            if ($LASTEXITCODE -ne 0) {
                Fail "重试仍失败。请手动执行：conda config --remove-key channels"
            }
        }
        Good "环境创建完成"
    }
}

# ---------- 4. 装 torch ----------
if ($hasNvidia) {
    $torchIndex = "https://download.pytorch.org/whl/" + $Index
    Info ("PyTorch: CUDA " + $Index)
} else {
    $torchIndex = "https://download.pytorch.org/whl/cpu"
    Info "PyTorch: CPU 版"
}

$pyBase = @("run", "-n", $EnvName, "python")

if ($DryRun) {
    Plan ("conda " + ($pyBase -join " ") + " -m pip install --upgrade pip")
    Plan ("conda " + ($pyBase -join " ") + " -m pip install torch torchvision --index-url " + $torchIndex)
} else {
    & $condaBin @pyBase -m pip install --upgrade pip
    & $condaBin @pyBase -m pip install torch torchvision --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) {
        Fail "PyTorch 安装失败。检查网络，或改用 -Index cu128 / -Index cu126"
    }
    Good "PyTorch 安装完成"
}

# ---------- 5. 装其余依赖 ----------
$reqFile = Join-Path $PSScriptRoot "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Fail "找不到 requirements.txt，应与本脚本在同一目录"
}

if ($DryRun) {
    Plan ("conda " + ($pyBase -join " ") + " -m pip install -r " + $reqFile)
} else {
    Info "安装其余依赖"
    & $condaBin @pyBase -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) { Fail "依赖安装失败" }
    Good "依赖安装完成"
}

# ---------- 6. 验证 ----------
$verifyFile = Join-Path $PSScriptRoot "verify_env.py"
if ($DryRun) {
    Plan ("conda " + ($pyBase -join " ") + " " + $verifyFile)
} elseif (Test-Path $verifyFile) {
    Info "验证环境"
    & $condaBin @pyBase $verifyFile
    if ($LASTEXITCODE -ne 0) { Fail "验证未通过，见上面输出" }
} else {
    Warn "未找到 verify_env.py，跳过验证"
}

Write-Host ""
if ($DryRun) {
    Good "dry-run 结束，没有改动任何东西。去掉 -DryRun 即可真正安装。"
} else {
    Good ("全部完成。激活环境: conda activate " + $EnvName)
    Write-Host "    下一步：按 环境迁移说明.md 重新下载模型和数据集" -ForegroundColor DarkGray
}
