"""Windows 一键安装脚本（国内镜像加速）
用法: python windows_setup_install_cn.py
"""
import os
import subprocess
import sys

MIRROR = "https://npmmirror.com/mirrors/playwright"

def run(cmd: str, **kwargs) -> int:
    print(f"  > {cmd}")
    return subprocess.run(cmd, shell=True, check=False, **kwargs).returncode

def main():
    print("=" * 48)
    print("  搜索引擎聚合爬虫 — Windows 环境安装")
    print("=" * 48)

    os.environ["PLAYWRIGHT_DOWNLOAD_HOST"] = MIRROR
    print(f"\n[1/3] 已设置镜像: {MIRROR}")

    print("\n[2/3] 安装 Python 依赖...")
    if run(f"{sys.executable} -m pip install -r requirements.txt") != 0:
        print("⚠ 依赖安装失败，请检查网络或手动执行 pip install -r requirements.txt")
        sys.exit(1)

    print("\n[3/3] 安装 Chromium 浏览器...")
    if run(f"{sys.executable} -m playwright install chromium") != 0:
        print("⚠ Chromium 安装失败，请手动执行 playwright install chromium")
        sys.exit(1)

    print()
    print("=" * 48)
    print("  安装完成！运行: python request_webs.py")
    print("=" * 48)


if __name__ == "__main__":
    main()
