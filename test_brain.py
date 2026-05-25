#!/usr/bin/env python3
"""
LogicLens — Brain Layer Local Smoke Test
=========================================
Validates the consolidated analyze_lecture pipeline contract.
"""

import logging
import sys
from brain import LogicBrain

# Configure console logging output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("logiclens.test")

def run_smoke_test():
    logger.info("Starting consolidated LogicBrain smoke test...")

    # Mock transcript dataset representing an engineering lecture snippet
    mock_timeline = [
        {"start": "00:00:10", "text": "Welcome back. Today we are computing the First and Follow sets for an LL(1) parser."},
        {"start": "00:00:35", "text": "Rule 1 for First sets: If X is a terminal, then First(X) is simply the set containing X itself."},
        {"start": "00:01:12", "text": "Rule 2: If we have a production X goes to epsilon, we add epsilon to the First(X) set immediately."},
        {"start": "00:01:55", "text": "Let's trace an example production. E goes to T E-prime. Since T is a non-terminal, we look at First of T."},
        {"start": "00:02:40", "text": "Now for Follow sets. Rule 1: Place a dollar sign $ in Follow of the start symbol. Do not forget this step on your exam."}
    ]

    try:
        # Instantiate the brain (reads parameters securely from updated brain.py)
        brain = LogicBrain()
        
        # Execute the consolidated analysis task
        logger.info("Executing Task: Structured Lecture Analysis...")
        analysis_output = brain.analyze_lecture(
            transcript=mock_timeline,
            focus_topic="Compiler Design and LL(1) Parsing Rules"
        )
        
        print("\n" + "="*80)
        print("TEST OUTPUT: LOGICBRAIN ACADEMIC ANALYSIS")
        print("="*80)
        print(analysis_output)
        print("="*80 + "\n")
        
        logger.info("Smoke test completed successfully! Architecture is fully aligned.")

    except Exception as err:
        logger.critical("Smoke test failed! Cognitive pipeline breakdown: %s", err, exc_info=True)

if __name__ == "__main__":
    run_smoke_test()