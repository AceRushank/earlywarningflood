#!/usr/bin/env python3
"""
scripts/download_mosdac_hdf.py

MOSDAC INSAT-3DS HDF5 Archive SFTP Discovery & Download Pipeline.
Connects via SFTP to download.mosdac.gov.in, discovers remote order files,
filters for 3SIMG_*_L2B_HEM_V01R00.h5 between 08-SEP-2026 and 22-SEP-2026 inclusive,
and downloads them safely with resume/skip support into data/mosdac/hdf_archive/.
"""

import os
import re
import sys
import stat
import time
import argparse
import getpass
from pathlib import Path
from datetime import datetime, date
from typing import List, Tuple, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import paramiko


SFTP_HOST = "download.mosdac.gov.in"
SFTP_PORT = 22

TARGET_START_DATE = date(2026, 9, 8)
TARGET_END_DATE = date(2026, 9, 22)

FILENAME_PATTERN = re.compile(r"^3SIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_L2B_HEM_V01R00\.h5$")


def get_credentials(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Securely retrieves credentials from environment variables, CLI, or interactive prompt.
    Never prints or logs the password.
    """
    username = (
        args.username or
        os.environ.get("MOSDAC_USERNAME") or
        os.environ.get("MOSDAC_USER")
    )
    password = (
        args.password or
        os.environ.get("MOSDAC_PASSWORD") or
        os.environ.get("MOSDAC_PASS")
    )

    if not username:
        if sys.stdin.isatty():
            username = input("Enter MOSDAC SFTP Username: ").strip()
        else:
            print("ERROR: MOSDAC username not provided. Set MOSDAC_USERNAME in .env or environment.")
            sys.exit(1)

    if not password:
        if sys.stdin.isatty():
            password = getpass.getpass("Enter MOSDAC SFTP Password: ").strip()
        else:
            print("ERROR: MOSDAC password not provided. Set MOSDAC_PASSWORD in .env or environment.")
            sys.exit(1)

    return username, password


def is_in_date_range(filename: str, start_dt: date, end_dt: date) -> bool:
    """Checks if filename date falls within [start_dt, end_dt] inclusive."""
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return False
    date_str = match.group(1)
    try:
        file_date = datetime.strptime(date_str, "%d%b%Y").date()
        return start_dt <= file_date <= end_dt
    except ValueError:
        return False


def connect_sftp(username: str, password: str, max_retries: int = 5) -> Tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    """Connects to MOSDAC SFTP server with robust retry and banner timeout."""
    for attempt in range(1, max_retries + 1):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=SFTP_HOST,
                port=SFTP_PORT,
                username=username,
                password=password,
                timeout=45,
                banner_timeout=60,
                auth_timeout=60,
                look_for_keys=False,
                allow_agent=False,
            )
            sftp = ssh.open_sftp()
            return ssh, sftp
        except Exception as e:
            if attempt == max_retries:
                raise e
            wait_time = attempt * 3
            print(f"Connection attempt {attempt} failed ({e}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError("Could not connect to SFTP server after retries")


def discover_remote_files(
    sftp: paramiko.SFTPClient,
    current_dir: str = ".",
    visited: set = None
) -> Dict[str, Tuple[str, int]]:
    """
    Recursively discovers all matching HDF5 files in the remote directory tree.
    Returns dictionary mapping unique filename -> (remote_full_path, size_bytes).
    Deduplicates files if present in multiple order directories.
    """
    if visited is None:
        visited = set()

    normalized_dir = current_dir.replace("\\", "/")
    if normalized_dir in visited:
        return {}
    visited.add(normalized_dir)

    results: Dict[str, Tuple[str, int]] = {}

    try:
        entries = sftp.listdir_attr(current_dir)
    except Exception as e:
        print(f"Warning: Could not list directory {current_dir}: {e}")
        return {}

    for entry in entries:
        remote_path = f"{current_dir}/{entry.filename}".replace("//", "/")
        if stat.S_ISDIR(entry.st_mode):
            sub_results = discover_remote_files(sftp, remote_path, visited)
            for fname, val in sub_results.items():
                if fname not in results:
                    results[fname] = val
        elif stat.S_ISREG(entry.st_mode):
            fname = entry.filename
            if is_in_date_range(fname, TARGET_START_DATE, TARGET_END_DATE):
                if fname not in results:
                    results[fname] = (remote_path, entry.st_size)

    return results


def download_file_with_retry(
    sftp_holder: list,
    username: str,
    password: str,
    remote_path: str,
    local_path: Path,
    expected_size: int,
    max_retries: int = 3
) -> str:
    """
    Downloads a single file safely with automatic session recovery and atomic rename.
    """
    if local_path.exists() and local_path.stat().st_size == expected_size and expected_size > 0:
        return "skipped"

    tmp_path = local_path.with_suffix(".tmp_download")

    for attempt in range(1, max_retries + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()

            sftp = sftp_holder[1]
            sftp.get(remote_path, str(tmp_path))

            downloaded_size = tmp_path.stat().st_size
            if expected_size > 0 and downloaded_size != expected_size:
                print(f"Warning: Size mismatch on attempt {attempt} for {local_path.name} (got {downloaded_size}, expected {expected_size})")
                tmp_path.unlink(missing_ok=True)
                if attempt == max_retries:
                    return "failed"
                time.sleep(2)
                continue

            tmp_path.replace(local_path)
            return "downloaded"

        except Exception as e:
            print(f"Warning: Download attempt {attempt} failed for {local_path.name}: {e}")
            tmp_path.unlink(missing_ok=True)
            if attempt == max_retries:
                return "failed"
            # Reconnect SFTP session if connection was severed
            try:
                print("Re-establishing SFTP session...")
                sftp_holder[1].close()
                sftp_holder[0].close()
            except Exception:
                pass
            time.sleep(attempt * 3)
            try:
                ssh, sftp = connect_sftp(username, password)
                sftp_holder[0] = ssh
                sftp_holder[1] = sftp
            except Exception as conn_err:
                print(f"Reconnection failed: {conn_err}")

    return "failed"


def main():
    parser = argparse.ArgumentParser(description="Download MOSDAC INSAT-3DS HDF5 archive via SFTP.")
    parser.add_argument("--username", help="MOSDAC SFTP username (default: MOSDAC_USERNAME env var)")
    parser.add_argument("--password", help="MOSDAC SFTP password (default: MOSDAC_PASSWORD env var)")
    parser.add_argument("--remote-dir", default=".", help="Remote base directory to search (default: .)")
    parser.add_argument("--dest-dir", default="data/mosdac/hdf_archive", help="Local destination directory")
    parser.add_argument("--dry-run", action="store_true", help="Discover and report remote files without downloading")
    parser.add_argument("--limit", type=int, help="Optional limit on number of files to download (for testing)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    dest_path = (base_dir / args.dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    username, password = get_credentials(args)

    print("=" * 80)
    print("MOSDAC INSAT-3DS HDF5 ARCHIVE SFTP DOWNLOAD")
    print(f"Host:       {SFTP_HOST}:{SFTP_PORT}")
    print(f"User:       {username}")
    print(f"Product:    3SIMG_L2B_HEM (HDF5)")
    print(f"Date range: {TARGET_START_DATE.strftime('%d-%b-%Y')} to {TARGET_END_DATE.strftime('%d-%b-%Y')} inclusive")
    print(f"Local dest: {dest_path}")
    print("=" * 80)

    print(f"Connecting to {SFTP_HOST}...")
    ssh, sftp = connect_sftp(username, password)
    sftp_holder = [ssh, sftp]
    print("Connected successfully to MOSDAC SFTP server!")

    try:
        print(f"Scanning remote directories starting at '{args.remote_dir}'...")
        file_map = discover_remote_files(sftp_holder[1], args.remote_dir)

        # Sort filenames chronologically
        sorted_filenames = sorted(file_map.keys())
        total_found = len(sorted_filenames)

        print("=" * 80)
        print(f"DISCOVERY RESULTS:")
        print(f"Found {total_found} unique matching HDF5 files for target date range ({TARGET_START_DATE} to {TARGET_END_DATE})")
        if sorted_filenames:
            print(f"Earliest:   {sorted_filenames[0]}")
            print(f"Latest:     {sorted_filenames[-1]}")
        print("=" * 80)

        if total_found == 0:
            print("No matching files found. Check your MOSDAC order directory or remote path.")
            return

        if args.dry_run:
            print("[Dry run mode] Skipping download. Sample of files found:")
            for fname in sorted_filenames[:10]:
                rpath, size = file_map[fname]
                print(f"   {fname} ({size / (1024*1024):.2f} MB)")
            if total_found > 10:
                print(f"   ... and {total_found - 10} more files.")
            return

        if args.limit:
            sorted_filenames = sorted_filenames[:args.limit]
            print(f"Limiting download to first {args.limit} files as requested.")

        downloaded_count = 0
        skipped_count = 0
        failed_count = 0

        t_start = time.time()
        for idx, fname in enumerate(sorted_filenames, 1):
            remote_path, size = file_map[fname]
            local_file = dest_path / fname
            status = download_file_with_retry(sftp_holder, username, password, remote_path, local_file, size)
            if status == "downloaded":
                downloaded_count += 1
                if idx % 10 == 0 or idx == len(sorted_filenames) or idx <= 5:
                    elapsed = time.time() - t_start
                    rate = (downloaded_count * 9.3) / max(1, elapsed)
                    print(f"[{idx:03d}/{len(sorted_filenames):03d}] [DOWNLOADED] {fname} ({size / (1024*1024):.2f} MB, ~{rate:.2f} MB/s)")
            elif status == "skipped":
                skipped_count += 1
                if idx % 50 == 0 or idx == len(sorted_filenames) or idx <= 5:
                    print(f"[{idx:03d}/{len(sorted_filenames):03d}] [SKIPPED]    {fname} (already exists)")
            else:
                failed_count += 1
                print(f"[{idx:03d}/{len(sorted_filenames):03d}] [FAILED]     {fname}")

        t_total = time.time() - t_start
        print("=" * 80)
        print("DOWNLOAD SUMMARY:")
        print(f"   Total matching files:       {len(sorted_filenames)}")
        print(f"   Downloaded:                 {downloaded_count}")
        print(f"   Skipped (already present):  {skipped_count}")
        print(f"   Failed:                     {failed_count}")
        print(f"   Total time:                 {t_total:.1f}s (~{t_total/60:.2f} min)")
        print("=" * 80)

    finally:
        try:
            sftp_holder[1].close()
            sftp_holder[0].close()
        except Exception:
            pass
        print("SFTP session closed.")


if __name__ == "__main__":
    main()
