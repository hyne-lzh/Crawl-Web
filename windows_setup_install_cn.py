import os
os.system("pip install -r requirements.txt")
os.system("set PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright")
os.system("playwright install chromium")