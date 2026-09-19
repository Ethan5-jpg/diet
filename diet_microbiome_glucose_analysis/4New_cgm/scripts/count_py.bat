@echo off
setlocal enabledelayedexpansion

echo Current directory: %cd%
echo =============================================

set count=0
set total=0

for %%f in (*.py) do (
    set /a count+=1

    for /f %%a in ('find /v /c "" ^< "%%f"') do (
        set lines=%%a
    )

    echo %%f    !lines! lines
    set /a total+=!lines!
)

echo =============================================
echo Python file count: %count%
echo Total lines: %total%

pause