"""
fetch_pecanstreet.py
--------------------
Connect to the Pecan Street Dataport PostgreSQL database, pull a small sample
of residential solar generation data (a few Austin homes, a few days in 2022),
and save to data/raw/pecanstreet_sample.csv.

Pecan Street Dataport is a direct PostgreSQL database, NOT a REST API.
Connection parameters (host, port, database name) are shown on
https://dataport.pecanstreet.org/access after logging in.

Usage:
    python src/fetch_pecanstreet.py

Requires in .env:
    PECAN_STREET_USERNAME    — your Dataport email address
    PECAN_STREET_PASSWORD    — your Dataport password
    PECAN_STREET_DB_HOST     — hostname from dataport.pecanstreet.org/access
    PECAN_STREET_DB_PORT     — port from dataport.pecanstreet.org/access
    PECAN_STREET_DB_NAME     — database name from dataport.pecanstreet.org/access

Install:  pip install psycopg2-binary python-dotenv polars
"""

import os
import sys

import polars as pl
from dotenv import load_dotenv

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    sys.exit(
        "ERROR: psycopg2 is not installed.\n"
        "  Run:  pip install psycopg2-binary"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

USERNAME = os.getenv("PECAN_STREET_USERNAME")
PASSWORD = os.getenv("PECAN_STREET_PASSWORD")
DB_HOST  = os.getenv("PECAN_STREET_DB_HOST")
DB_PORT  = os.getenv("PECAN_STREET_DB_PORT", "5432")
DB_NAME  = os.getenv("PECAN_STREET_DB_NAME")

_MISSING = []
if not USERNAME or USERNAME == "your_username_here":
    _MISSING.append("PECAN_STREET_USERNAME")
if not PASSWORD or PASSWORD == "your_password_here":
    _MISSING.append("PECAN_STREET_PASSWORD")
if not DB_HOST or DB_HOST == "your_db_host_here":
    _MISSING.append("PECAN_STREET_DB_HOST  (from dataport.pecanstreet.org/access)")
if not DB_NAME or DB_NAME == "your_db_name_here":
    _MISSING.append("PECAN_STREET_DB_NAME  (from dataport.pecanstreet.org/access)")
if _MISSING:
    sys.exit(
        "ERROR: The following .env variables are missing or still set to placeholders:\n"
        + "\n".join(f"  - {v}" for v in _MISSING)
        + "\n\nVisit https://dataport.pecanstreet.org/access (after logging in) "
          "to find the PostgreSQL host, port, and database name."
    )

# Sample window
START_DATE = "2022-06-01"
END_DATE   = "2022-06-03"   # exclusive upper bound
MAX_HOMES  = 5

OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(OUT_DIR, "pecanstreet_sample.csv")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_connection() -> "psycopg2.connection":
    """Open and return a psycopg2 connection to the Dataport PostgreSQL DB."""
    print(f"Connecting to Dataport PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME} ...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            dbname=DB_NAME,
            user=USERNAME,
            password=PASSWORD,
            connect_timeout=15,
            sslmode="require",
        )
        print("Connected successfully.")
        return conn
    except psycopg2.OperationalError as exc:
        msg = str(exc).lower()
        if "password authentication" in msg or "authentication failed" in msg:
            sys.exit(
                "ERROR: Authentication failed. "
                "Check PECAN_STREET_USERNAME and PECAN_STREET_PASSWORD in .env."
            )
        if "could not connect" in msg or "connection refused" in msg or "timeout" in msg:
            sys.exit(
                f"ERROR: Could not reach {DB_HOST}:{DB_PORT}. "
                "Check PECAN_STREET_DB_HOST and PECAN_STREET_DB_PORT in .env."
            )
        sys.exit(f"ERROR: Database connection failed:\n  {exc}")


def fetch_sample(conn) -> pl.DataFrame:
    """Pull 15-minute solar generation data from electricity.eg_realpower_15min.

    Schema (from official Pecan Street DataPort examples):
        electricity.eg_realpower_15min
            dataid      INT   — unique home identifier
            local_15min TIMESTAMP — local timestamp (15-min intervals)
            solar       FLOAT — solar generation in kWh per 15-min interval

        other_datasets.metadata
            dataid, city, solar (not null = has solar panels), data availability

    Args:
        conn: Open psycopg2 database connection.

    Returns:
        Polars DataFrame with columns ``dataid``, ``local_15min``, ``solar``.
    """
    print(f"Finding up to {MAX_HOMES} Austin homes with solar data ...")

    id_query = """
        SELECT dataid
        FROM other_datasets.metadata
        WHERE city = 'Austin'
          AND solar IS NOT NULL
          AND egauge_1min_min_time < %(start)s
          AND egauge_1min_max_time >= %(end)s
        LIMIT %(limit)s;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(id_query, {"start": START_DATE, "end": END_DATE, "limit": MAX_HOMES})
        rows = cur.fetchall()

    if not rows:
        sys.exit(
            "ERROR: No Austin homes with solar data found in the queried window. "
            "Try adjusting START_DATE / END_DATE."
        )

    dataids = [r["dataid"] for r in rows]
    print(f"  Home IDs: {dataids}")

    print(f"Fetching 15-min solar data for {START_DATE} → {END_DATE} ...")
    data_query = """
        SELECT dataid, local_15min, solar
        FROM electricity.eg_realpower_15min
        WHERE dataid = ANY(%(ids)s)
          AND local_15min >= %(start)s
          AND local_15min  < %(end)s
        ORDER BY dataid, local_15min;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(data_query, {
            "ids":   dataids,
            "start": START_DATE,
            "end":   END_DATE,
        })
        records = cur.fetchall()

    if not records:
        sys.exit("ERROR: Query returned 0 rows. Check date range or column names.")

    return pl.from_dicts([dict(r) for r in records])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = get_connection()
    try:
        df = fetch_sample(conn)
    finally:
        conn.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    df.write_csv(OUT_PATH)
    print(f"\nSaved {len(df):,} rows to {OUT_PATH}")
    print(df.head(10))


if __name__ == "__main__":
    main()
