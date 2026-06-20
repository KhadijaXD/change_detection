"""
Sentinel-2 Data Downloader
===========================
Downloads Sentinel-2 L2A imagery from the Copernicus Data Space using
the sentinelhub Python library.

Pre-configured regions:
  1. Lahore urban periphery (2019 vs 2024)
  2. 2022 Indus River flood region (Sindh / South Punjab)

Usage:
    python download_sentinel.py --region lahore --year 2019
    python download_sentinel.py --region lahore --year 2024
    python download_sentinel.py --region floods --year 2022 --month 8

Before running, you need a free Copernicus Data Space account.
See instructions printed when running this script.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# --- REGION BOUNDING BOXES ---
REGIONS = {
    "lahore": {
        "bbox": [74.20, 31.35, 74.45, 31.60],  # [lon_min, lat_min, lon_max, lat_max]
        "description": "Lahore urban periphery",
    },
    "floods": {
        "bbox": [68.0, 27.0, 70.0, 28.5],
        "description": "2022 Indus flood region (Sindh / South Punjab)",
    },
}


def print_setup_instructions():
    """Print instructions for obtaining Copernicus credentials."""
    instructions = """
╔══════════════════════════════════════════════════════════════════════╗
║              COPERNICUS DATA SPACE — SETUP INSTRUCTIONS            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. Create a FREE account at:                                      ║
║     https://dataspace.copernicus.eu/                               ║
║                                                                    ║
║  2. After registering, go to your dashboard and create an          ║
║     OAuth client under "Account Settings → OAuth Clients".         ║
║                                                                    ║
║  3. Set these environment variables before running this script:    ║
║                                                                    ║
║     export SH_CLIENT_ID="your-client-id"                           ║
║     export SH_CLIENT_SECRET="your-client-secret"                   ║
║                                                                    ║
║  4. Alternatively, create a file ~/.sentinelhub/config.toml:       ║
║                                                                    ║
║     [default]                                                      ║
║     sh_client_id = "your-client-id"                                ║
║     sh_client_secret = "your-client-secret"                        ║
║     sh_base_url = "https://sh.dataspace.copernicus.eu"             ║
║     sh_token_url = "https://identity.dataspace.copernicus.eu/...   ║
║                    auth/realms/CDSE/protocol/openid-connect/token" ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)


def download_sentinel_patch(
    bbox: list,
    time_start: str,
    time_end: str,
    output_dir: str,
    resolution: int = 10,
    max_cloud: float = 20.0,
):
    """Download a Sentinel-2 L2A RGB patch.

    Args:
        bbox: [lon_min, lat_min, lon_max, lat_max].
        time_start: Start date 'YYYY-MM-DD'.
        time_end: End date 'YYYY-MM-DD'.
        output_dir: Directory to save outputs.
        resolution: Spatial resolution in meters (default 10m).
        max_cloud: Maximum cloud coverage percentage.
    """
    try:
        from sentinelhub import (
            SHConfig,
            BBox,
            CRS,
            DataCollection,
            MimeType,
            SentinelHubRequest,
            bbox_to_dimensions,
        )
    except ImportError:
        print("[ERROR] sentinelhub is not installed. Run: pip install sentinelhub")
        sys.exit(1)

    # --- CONFIGURE ---
    config = SHConfig()

    # Check for env vars
    client_id = os.environ.get("SH_CLIENT_ID", "")
    client_secret = os.environ.get("SH_CLIENT_SECRET", "")
    if client_id and client_secret:
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret
        config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        config.sh_token_url = (
            "https://identity.dataspace.copernicus.eu/"
            "auth/realms/CDSE/protocol/openid-connect/token"
        )

    if not config.sh_client_id or not config.sh_client_secret:
        print("[ERROR] Sentinel Hub credentials not configured.")
        print_setup_instructions()
        sys.exit(1)

    # --- EVALSCRIPT: TRUE COLOR RGB (B04, B03, B02) ---
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B04", "B03", "B02"],
                units: "DN"
            }],
            output: {
                bands: 3,
                sampleType: "INT16"
            }
        };
    }
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """

    roi = BBox(bbox=bbox, crs=CRS.WGS84)
    size = bbox_to_dimensions(roi, resolution=resolution)
    print(f"[INFO] Image dimensions at {resolution}m: {size[0]} × {size[1]} pixels")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    "s2l2a", service_url=config.sh_base_url
                ),
                time_interval=(time_start, time_end),
                maxcc=max_cloud / 100.0,
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF),
        ],
        bbox=roi,
        size=size,
        config=config,
    )

    print(f"[INFO] Requesting data for {time_start} to {time_end}...")
    data = request.get_data()

    if not data or len(data) == 0:
        print("[WARN] No data returned. Try expanding the date range or cloud threshold.")
        return

    img = data[0]
    print(f"[INFO] Received image shape: {img.shape}, dtype: {img.dtype}")

    # --- SAVE AS GEOTIFF ---
    try:
        import rasterio
        from rasterio.transform import from_bounds

        tiff_path = output_path / f"sentinel2_{time_start}_{time_end}.tif"
        transform = from_bounds(*bbox, img.shape[1], img.shape[0])

        with rasterio.open(
            str(tiff_path), "w",
            driver="GTiff",
            height=img.shape[0], width=img.shape[1], count=3,
            dtype=img.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            for band_idx in range(3):
                dst.write(img[:, :, band_idx], band_idx + 1)
        print(f"[INFO] GeoTIFF saved: {tiff_path}")
    except ImportError:
        print("[WARN] rasterio not installed — skipping GeoTIFF save.")

    # --- SAVE NORMALIZED PNG ---
    from PIL import Image as PILImage

    img_float = img.astype(np.float32)
    # Clip to typical Sentinel-2 reflectance range and normalize to 0-255
    img_clipped = np.clip(img_float / 3000.0, 0, 1)
    img_uint8 = (img_clipped * 255).astype(np.uint8)

    png_path = output_path / f"sentinel2_{time_start}_{time_end}.png"
    PILImage.fromarray(img_uint8).save(str(png_path))
    print(f"[INFO] PNG saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-2 imagery")
    parser.add_argument(
        "--region", type=str, required=True, choices=list(REGIONS.keys()),
        help="Pre-configured region name.",
    )
    parser.add_argument("--year", type=int, required=True, help="Year to download.")
    parser.add_argument("--month", type=int, default=None, help="Specific month (1-12). If omitted, downloads best available for the year.")
    parser.add_argument("--output_dir", type=str, default="./sentinel_data")
    parser.add_argument("--resolution", type=int, default=10, help="Resolution in meters.")
    parser.add_argument("--max_cloud", type=float, default=20.0, help="Max cloud cover %%.")
    args = parser.parse_args()

    region = REGIONS[args.region]
    print(f"[INFO] Region: {region['description']}")
    print(f"[INFO] Bounding box: {region['bbox']}")

    print_setup_instructions()

    # Build date range
    if args.month:
        # Specific month
        start = f"{args.year}-{args.month:02d}-01"
        # End of month approximation
        if args.month == 12:
            end = f"{args.year}-12-31"
        else:
            end = f"{args.year}-{args.month + 1:02d}-01"
    else:
        # Best from entire year — use dry season months for cleaner imagery
        start = f"{args.year}-01-01"
        end = f"{args.year}-12-31"

    out_dir = os.path.join(args.output_dir, args.region, str(args.year))

    download_sentinel_patch(
        bbox=region["bbox"],
        time_start=start,
        time_end=end,
        output_dir=out_dir,
        resolution=args.resolution,
        max_cloud=args.max_cloud,
    )


if __name__ == "__main__":
    main()
