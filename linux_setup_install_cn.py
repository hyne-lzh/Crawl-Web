"""Linux 一键安装脚本（国内镜像加速，支持 Debian/Ubuntu/Fedora/Arch）
用法: python linux_setup_install_cn.py
"""
import os
import shutil
import subprocess
import sys

MIRROR = "https://npmmirror.com/mirrors/playwright"

DEBIAN_DEPS = [
    "libnss3", "libnspr4", "libatk-bridge2.0-0", "libdrm2",
    "libxkbcommon0", "libatspi2.0-0", "libxcomposite1",
    "libxdamage1", "libxfixes3", "libxrandr2", "libgbm1",
    "libpango-1.0-0", "libcairo2", "libasound2",
]

FEDORA_DEPS = [
    "nss", "nspr", "atk", "at-spi2-atk", "cups-libs", "libdrm",
    "libxkbcommon", "libXcomposite", "libXdamage", "libXrandr",
    "mesa-libgbm", "pango", "cairo", "alsa-lib",
]

ARCH_DEPS = [
    "nss", "nspr", "atk", "at-spi2-atk", "libdrm",
    "libxkbcommon", "libxcomposite", "libxdamage", "libxrandr",
    "mesa", "pango", "cairo", "alsa-lib",
]

def run(cmd: str, **kwargs) -> int:
    print(f"  > {cmd}")
    return subprocess.run(cmd, shell=True, check=False, **kwargs).returncode

def detect_pkg_manager() -> str | None:
    """检测系统包管理器"""
    if shutil.which("apt-get"):
        return "apt"
    if shutil.which("dnf"):
        return "dnf"
    if shutil.which("yum"):
        return "yum"
    if shutil.which("pacman"):
        return "pacman"
    return None

def install_system_deps() -> bool:
    """安装 Chromium 所需的系统依赖"""
    pkg = detect_pkg_manager()

    if pkg == "apt":
        print("  检测到 apt (Debian/Ubuntu)，安装系统依赖...")
        run("sudo apt-get update -qq")
        deps = " ".join(DEBIAN_DEPS)
        # 先尝试含 asound2t64 的新版包名，失败则用旧版
        return run(f"sudo apt-get install -y -qq {deps} libasound2t64 2>/dev/null || sudo apt-get install -y -qq {deps}") == 0

    elif pkg in ("dnf", "yum"):
        print(f"  检测到 {pkg} (Fedora/RHEL)，安装系统依赖...")
        deps = " ".join(FEDORA_DEPS)
        return run(f"sudo {pkg} install -y {deps}") == 0

    elif pkg == "pacman":
        print("  检测到 pacman (Arch)，安装系统依赖...")
        deps = " ".join(ARCH_DEPS)
        return run(f"sudo pacman -S --noconfirm {deps}") == 0

    else:
        print("  ⚠ 未检测到已知包管理器，跳过系统依赖安装。")
        print("    如 playwright 运行报错，请手动安装 Chromium 系统依赖。")
        return True

def main():
    print("=" * 48)
    print("  搜索引擎聚合爬虫 — Linux 环境安装")
    print("=" * 48)

    os.environ["PLAYWRIGHT_DOWNLOAD_HOST"] = MIRROR
    print(f"\n[1/4] 已设置镜像: {MIRROR}")

    print("\n[2/4] 安装 Chromium 系统依赖...")
    install_system_deps()

    print("\n[3/4] 安装 Python 依赖...")
    if run(f"{sys.executable} -m pip install -r requirements.txt") != 0:
        print("⚠ 依赖安装失败，请检查网络或手动执行 pip install -r requirements.txt")
        sys.exit(1)

    print("\n[4/4] 安装 Chromium 浏览器...")
    if run(f"{sys.executable} -m playwright install --with-deps chromium") != 0:
        print("⚠ Chromium 安装失败，请手动执行 playwright install --with-deps chromium")
        sys.exit(1)

    print()
    print("=" * 48)
    print("  安装完成！运行: python request_webs.py")
    print("=" * 48)


if __name__ == "__main__":
    main()
