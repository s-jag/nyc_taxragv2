#!/usr/bin/env python3
"""One-time script to upload vectors to Qdrant Cloud.

This script reads the existing docstore.json and uploads all vectors
to Qdrant Cloud. Run this once after setting up your Qdrant Cloud cluster.

Usage:
    export QDRANT_URL="https://xxx.aws.cloud.qdrant.io"
    export QDRANT_API_KEY="your-api-key"
    python scripts/upload_to_qdrant.py

Or with explicit arguments:
    python scripts/upload_to_qdrant.py --url "https://xxx.aws.cloud.qdrant.io" --api-key "your-key"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Upload vectors to Qdrant Cloud."""
    parser = argparse.ArgumentParser(
        description="Upload vectors to Qdrant Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        help="Qdrant Cloud URL (or set QDRANT_URL env var)",
        default=os.getenv("QDRANT_URL"),
    )
    parser.add_argument(
        "--api-key",
        help="Qdrant Cloud API key (or set QDRANT_API_KEY env var)",
        default=os.getenv("QDRANT_API_KEY"),
    )
    parser.add_argument(
        "--docstore",
        help="Path to docstore.json",
        default="data/docstore.json",
    )
    parser.add_argument(
        "--collection",
        help="Collection name in Qdrant",
        default="nyc_tax_law",
    )

    args = parser.parse_args()

    if not args.url:
        print("Error: QDRANT_URL not set. Use --url or set QDRANT_URL env var.")
        sys.exit(1)

    if not args.api_key:
        print("Error: QDRANT_API_KEY not set. Use --api-key or set QDRANT_API_KEY env var.")
        sys.exit(1)

    docstore_path = Path(args.docstore)
    if not docstore_path.exists():
        print(f"Error: Docstore not found at {docstore_path}")
        print("Run the ingestion pipeline first to create docstore.json")
        sys.exit(1)

    print("=" * 60)
    print("Qdrant Cloud Vector Upload")
    print("=" * 60)
    print(f"Qdrant URL: {args.url}")
    print(f"Collection: {args.collection}")
    print(f"Docstore: {docstore_path}")
    print("=" * 60)

    from app.ingest import VectorStoreManager

    manager = VectorStoreManager(
        use_qdrant=True,
        qdrant_url=args.url,
        qdrant_api_key=args.api_key,
    )

    print("\nStarting upload...")
    vectorstore = manager.create_vector_db(
        docstore_path=str(docstore_path),
        collection_name=args.collection,
    )

    print("\n" + "=" * 60)
    print("Upload complete!")
    print("=" * 60)
    print(f"\nYour vectors are now available at: {args.url}")
    print(f"Collection name: {args.collection}")
    print("\nNext steps:")
    print("1. Deploy to Streamlit Cloud")
    print("2. Add QDRANT_URL and QDRANT_API_KEY to Streamlit secrets")
    print("3. Add OPENAI_API_KEY to Streamlit secrets")


if __name__ == "__main__":
    main()
