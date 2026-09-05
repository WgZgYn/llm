# 编译 cuda_native/ 下的原生 CUDA C++ 示例
# 注意：Windows 上 nvcc 编译 host 代码需要 MSVC 的 cl.exe（Visual Studio Build Tools）。
#       若没有 MSVC（本机常见情况），请改在 Linux 上编译（见 cuda_native/Makefile）。
$ErrorActionPreference = "Stop"

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $nvcc) {
    if ($env:CUDA_PATH) { $nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe" }
}
if (-not $nvcc -or -not (Test-Path $nvcc)) {
    throw "未找到 nvcc，请先安装 CUDA Toolkit 并设置 CUDA_PATH。"
}
Write-Host "使用 nvcc: $nvcc"

# 检查 MSVC cl.exe（Windows 下必需，否则报 'Cannot find compiler cl.exe'）
$cl = Get-Command cl.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $cl) {
    throw @"
未找到 MSVC 编译器 cl.exe。
Windows 下 nvcc 需要 Visual Studio Build Tools（含 'Desktop development with C++'）。
替代方案：在 Linux 上运行 `make -C cuda_native` 编译这些 .cu 文件。
"@
}

$src = Join-Path $PSScriptRoot "cuda_native"
$out = Join-Path $PSScriptRoot "build"
New-Item -ItemType Directory -Force $out | Out-Null

$arch = "sm_89"   # RTX 4060 Laptop = Ada, compute capability 8.9

$files = Get-ChildItem $src -Filter "*.cu" | Sort-Object Name
foreach ($f in $files) {
    $exe = Join-Path $out ($f.BaseName + ".exe")
    Write-Host "编译 $($f.Name) -> $($f.BaseName).exe"
    & $nvcc "-arch=$arch" -O3 "-I$src" -o $exe $f.FullName
    if ($LASTEXITCODE -ne 0) { throw "编译失败: $($f.Name)" }
}
Write-Host "`n全部编译完成，输出目录: $out"
