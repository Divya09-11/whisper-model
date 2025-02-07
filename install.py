import subprocess
import sys

def install_package():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."]) 

if __name__ == "__main__":
    install_package()