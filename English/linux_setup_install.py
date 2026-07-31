"""Linux one-click setup script (supports Debian/Ubuntu/Fedora/Arch)
Usage: python linux_setup_install.py
"""
import shutil
import subprocess
import sys


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
    """Detect the system package manager"""
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
    """Install system dependencies required by Chromium"""
    pkg = detect_pkg_manager()

    if pkg == "apt":
        print("  Detected apt (Debian/Ubuntu), installing system dependencies...")
        run("sudo apt-get update -qq")
        deps = " ".join(DEBIAN_DEPS)
        return run(
            f"sudo apt-get install -y -qq {deps} libasound2t64 2>/dev/null"
            f" || sudo apt-get install -y -qq {deps}"
        ) == 0

    elif pkg in ("dnf", "yum"):
        print(f"  Detected {pkg} (Fedora/RHEL), installing system dependencies...")
        deps = " ".join(FEDORA_DEPS)
        return run(f"sudo {pkg} install -y {deps}") == 0

    elif pkg == "pacman":
        print("  Detected pacman (Arch), installing system dependencies...")
        deps = " ".join(ARCH_DEPS)
        return run(f"sudo pacman -S --noconfirm {deps}") == 0

    else:
        print("  WARNING: Unknown package manager, skipping system deps.")
        print("           If playwright fails, manually install Chromium system dependencies.")
        return True


def main():
    print("=" * 48)
    print("  Search Engine Aggregator — Linux Setup")
    print("=" * 48)

    print("\n[1/3] Installing Chromium system dependencies...")
    install_system_deps()

    print("\n[2/3] Installing Python dependencies...")
    if run(f"{sys.executable} -m pip install -r requirements.txt") != 0:
        print("ERROR: Dependency installation failed.")
        print("       Please check your network or run: pip install -r requirements.txt")
        sys.exit(1)

    print("\n[3/3] Installing Chromium browser...")
    if run(f"{sys.executable} -m playwright install --with-deps chromium") != 0:
        print("ERROR: Chromium installation failed.")
        print("       Please run: playwright install --with-deps chromium")
        sys.exit(1)

    print()
    print("=" * 48)
    print("  Setup complete! Run: python request_webs.py")
    print("=" * 48)


if __name__ == "__main__":
    main()
