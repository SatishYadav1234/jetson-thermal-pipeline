@echo off
REM Windows Build Script
if not exist build mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
echo [INFO] Build completed successfully!
cd ..
