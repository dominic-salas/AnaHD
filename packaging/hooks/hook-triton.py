"""Triton finds its GPU backends by listing the triton/backends folder for compiler.py and
driver.py, and compiles kernels from its own source files, so its modules stay as .py files
on disk instead of inside the PyInstaller archive."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

module_collection_mode = "py"
hiddenimports = collect_submodules("triton")
datas = collect_data_files("triton", include_py_files=True) + copy_metadata("triton-windows")
