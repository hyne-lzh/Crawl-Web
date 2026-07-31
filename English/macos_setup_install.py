"""macOS one-click setup script
Usage: python macos_setup_install.py
"""
import subprocess
import sys


def run(cmd: str, **kwargs) -> int:
    print(f"  > {cmd}")
    return subprocess.run(cmd, shell=True, check=False, **kwargs).returncode


def main():
    print("=" * 48)
    print("  Search Engine Aggregator — macOS Setup")
    print("=" * 48)

    print("\n[1/2] Installing Python dependencies...")
    if run(f"{sys.executable} -m pip install -r requirements.txt") != 0:
        print("ERROR: Dependency installation failed.")
        print("       Please check your network or run: pip install -r requirements.txt")
        sys.exit(1)

    print("\n[2/2] Installing Chromium browser...")
    if run(f"{sys.executable} -m playwright install chromium") != 0:
        print("ERROR: Chromium installation failed.")
        print("       Please run: playwright install chromium")
        sys.exit(1)

    print()
    print("=" * 48)
    print("  Setup complete! Run: python request_webs.py")
    print("=" * 48)


if __name__ == "__main__":
    main()
