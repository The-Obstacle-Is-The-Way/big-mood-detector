#!/usr/bin/env python3
"""
Enhanced health check script that verifies all dependencies.
"""

import os
import sys
import time
from urllib.request import urlopen


def check_api_health(port: int = 8000) -> bool:
    """Check if API is responding."""
    try:
        with urlopen(f"http://localhost:{port}/health", timeout=5) as response:
            return response.status == 200
    except Exception as e:
        print(f"API health check failed: {e}")
        return False


def check_database() -> bool:
    """Check database connectivity."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url or "postgresql" not in db_url:
        # Skip DB check if not configured
        return True

    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        conn.close()
        return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False


def check_redis() -> bool:
    """Check Redis connectivity."""
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        # Skip Redis check if not configured
        return True

    try:
        import redis
        client = redis.from_url(redis_url, socket_connect_timeout=5)
        client.ping()
        return True
    except Exception as e:
        print(f"Redis health check failed: {e}")
        return False


def main():
    """Run all health checks."""
    # Allow services time to start
    startup_delay = int(os.environ.get("HEALTH_CHECK_DELAY", "0"))
    if startup_delay > 0:
        time.sleep(startup_delay)

    checks = [
        ("API", check_api_health),
        ("Database", check_database),
        ("Redis", check_redis),
    ]

    all_healthy = True
    for name, check_func in checks:
        if check_func():
            print(f"✓ {name} is healthy")
        else:
            print(f"✗ {name} is unhealthy")
            all_healthy = False

    sys.exit(0 if all_healthy else 1)


if __name__ == "__main__":
    main()
