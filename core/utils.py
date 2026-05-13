import os
import sys
import logging
import subprocess

def check_permissions():
    """Ensure the script runs with administrative privileges."""
    if sys.platform.startswith("linux"):
        return os.geteuid() == 0
    return True  # Windows check requires separate logic

def get_nessus_www_path():
    """Identify the Nessus web directory based on the OS."""
    if sys.platform.startswith("linux"):
        return "/opt/nessus/var/nessus/www/"
    return r"C:\ProgramData\Tenable\Nessus\nessus\www"

def restart_nessus_service():
    """Execute service restart to apply changes."""
    logging.info("Restarting Nessus service...")
    try:
        if sys.platform.startswith("linux"):
            subprocess.run(["systemctl", "restart", "nessusd"], check=True)
        else:
            subprocess.run(["net", "stop", "Tenable Nessus"], check=False)
            subprocess.run(["net", "start", "Tenable Nessus"], check=True)
        logging.info("Service restarted successfully.")
    except Exception as e:
        logging.error(f"Could not restart service: {e}")
