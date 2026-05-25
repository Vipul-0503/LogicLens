"""
LogicLens Orchestration CLI Layer
=================================
Core runtime controller for LogicLens. Links engine processing streams 
with the Gemini cognitive layer, managing artifact serialization and 
downstream local delivery.

Usage:
    python main.py --input https://www.youtube.com/watch?v=VIDEO_ID --topic "Compiler Design"
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Module layout imports
from engine import TranscriptEngine
from brain import LogicBrain

# Configure structured runtime reporting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LogicLens.Main")


def parse_arguments() -> argparse.Namespace:
    """Parse down incoming terminal execution flags."""
    parser = argparse.ArgumentParser(
        description="LogicLens: High-Fidelity Timestamped Lecture Analysis Platform."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Target YouTube URL (or local video/JSON data trace)."
    )
    parser.add_argument(
        "-t", "--topic",
        type=str,
        default="General Computer Science",
        help="Academic target scope focus (e.g., 'Distributed Systems', 'Automata')."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="outputs",
        help="Local directory destination path for generated markdown artifacts."
    )
    parser.add_argument(
        "-c", "--cookies",
        type=str,
        default=None,
        help="Optional path to netscape cookies.txt file for authenticated extraction."
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrates pipeline serialization and file generation workflows."""
    args = parse_arguments()
    logger.info("Initializing LogicLens orchestration engine execution.")

    try:
        # 1. Determine if input is a URL or a fallback local path
        if args.input.startswith(("http://", "https://")):
            logger.info("Target input recognized as a live web stream URL.")
            
            # 2. Instantiate and trigger the real TranscriptEngine pipeline
            engine = TranscriptEngine(cookies_path=args.cookies)
            logger.info("Extracting raw VTT timeline from YouTube via yt-dlp...")
            transcript_timeline = engine.extract(args.input)
        else:
            raise ValueError(
                f"Input source must be a valid YouTube stream URL. Received: '{args.input}'"
            )

        # 3. Instantiate Cognitive Brain Gateway
        brain_layer = LogicBrain()

        # 4. Trigger Model Token Dispatch Pipeline
        logger.info("Dispatching pipeline data stream to cognitive layer...")
        analysis_report = brain_layer.analyze_lecture(
            transcript=transcript_timeline,
            focus_topic=args.topic
        )

        # 5. Build Artifact Export Structure
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Build clean filename template tokens using execution metadata
        safe_topic_str = args.topic.lower().replace(" ", "_").replace("/", "-")
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_filename = out_path / f"analysis_{safe_topic_str}_{timestamp_slug}.md"

        # 6. Persistent Local Delivery Write out
        logger.info("Writing structured academic report file...")
        with open(target_filename, "w", encoding="utf-8") as file:
            file.write(f"# LogicLens Analysis Report\n")
            file.write(f"**Target Topic Focus:** {args.topic}  \n")
            file.write(f"**Source Stream Link:** {args.input}  \n")
            file.write(f"**Generated Pipeline Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            file.write(analysis_report)

        print("\n" + "═"*80)
        print(f" SUCCESS: System Analysis Export Generated Cleanly.")
        print(f" Artifact Path Location: {target_filename.resolve()}")
        print("═"*80 + "\n")

    except Exception as pipeline_err:
        logger.critical("Fatal orchestration layer failure: %s", pipeline_err, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()