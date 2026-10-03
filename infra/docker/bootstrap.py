"""Wait for the Qdrant service, then migrate the separate Compose database."""

import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def wait_for_qdrant(url="http://qdrant:6333/readyz", attempts=60):
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        if attempt + 1 < attempts:
            time.sleep(1)
    raise RuntimeError("Qdrant did not become ready; inspect the qdrant service logs")


def main():
    wait_for_qdrant()
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)


if __name__ == "__main__":
    main()
