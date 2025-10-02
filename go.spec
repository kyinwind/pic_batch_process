# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['go.py'],  # 你的主程序文件
    pathex=[],  # 如果有额外路径可以添加，如：['./src']
    binaries=[],
    datas=[],  # 如果有数据文件需要打包，格式为：[('源路径', '目标路径')]
    hiddenimports=[],  # 手动指定PyInstaller可能漏掉的导入
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],  # 排除不需要的模块，减小体积
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,  # 关键：包含二进制文件
    a.zipfiles,  # 关键：包含压缩文件
    a.datas,     # 关键：包含数据文件
    [],
    exclude_binaries=False,
    name='go',  # 生成的EXE文件名
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # 启用UPX压缩（减小体积）
    console=False,  # 如果是GUI程序设为False，命令行程序设为True
    disable_windowed_traceback=False,
    target_arch=None,
    icon=None,  # 可选：添加图标，格式为'icon.ico'
    runtime_tmpdir=None,  # 单文件模式下临时目录自动处理
    standalone=True,  # 关键：明确指定单文件模式
    append_pkg=False,
)
    