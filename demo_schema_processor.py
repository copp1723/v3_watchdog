#!/usr/bin/env python3
"""
Demo script for Watchdog AI Semantic Layer & Business Guardrails.

This script demonstrates the complete semantic layer with:
1. Schema profile loading from disk
2. Business rule loading from YAML
3. Query rewriting for semantic correctness
4. Precision scoring for reliability prediction
"""

import os
import sys
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import our components
from src.utils.adaptive_schema import (
    SchemaProfile, SchemaColumn, ExecRole, SchemaProfileManager
)
from src.rule_engine import BusinessRuleEngine, init_rule_registry
from src.query_rewriter import QueryRewriter, QueryRewriteStats
from src.precision_scoring import PrecisionScoringEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("demo_schema_processor")

def load_components():
    """Load all system components."""
    logger.info("Loading components...")
    
    # 1. Initialize schema profiles from disk
    profile_manager = SchemaProfileManager("profiles")
    profiles = profile_manager.load_profiles()
    
    if not profiles:
        logger.error("No schema profiles found! Please run setup.py first.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(profiles)} schema profiles: {', '.join(profiles.keys())}")
    
    # 2. Initialize business rules from YAML
    rule_file = "BusinessRuleRegistry.yaml"
    if not os.path.exists(rule_file):
        logger.info(f"Rule registry not found, creating default at {rule_file}")
        init_rule_registry(rule_file)
    
    rule_engine = BusinessRuleEngine(rule_file)
    logger.info(f"Loaded {len(rule_engine.rules)} business rules")
    
    # 3. Create core components
    components = {
        "profile_manager": profile_manager,
        "rule_engine": rule_engine,
        "profiles": profiles
    }
    
    return components

def create_processors(components):
    """Create query processors for each profile."""
    processors = {}
    stats = QueryRewriteStats("query_stats.json")
    
    for profile_id, profile in components["profiles"].items():
        # Create query rewriter
        rewriter = QueryRewriter(
            schema_profile=profile,
            rule_engine=components["rule_engine"],
            stats=stats
        )
        
        # Create precision scorer
        scorer = PrecisionScoringEngine(profile)
        
        # Store processors
        processors[profile_id] = {
            "profile": profile,
            "rewriter": rewriter,
            "scorer": scorer
        }
        
        logger.info(f"Created processors for profile: {profile_id}")
    
    return processors

def process_query(query, profile_id, processors):
    """Process a query with a specific profile."""
    if profile_id not in processors:
        logger.error(f"Profile {profile_id} not found! Available profiles: {', '.join(processors.keys())}")
        return None
    
    logger.info(f"Processing query: '{query}' with profile: {profile_id}")
    
    # Get processors
    processor = processors[profile_id]
    rewriter = processor["rewriter"]
    scorer = processor["scorer"]
    
    # Rewrite query
    rewritten_query, metadata = rewriter.rewrite_query(query)
    
    # Score query precision
    precision = scorer.predict_precision(rewritten_query, metadata)
    
    # Prepare result
    result = {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "profile_id": profile_id,
        "profile_name": processor["profile"].name,
        "precision_score": precision["score"],
        "confidence_level": precision["confidence"],
        "reasoning": precision["reasoning"],
        "rewrites": metadata.get("rewrites", []),
        "column_matches": metadata.get("column_matches", {}),
        "missing_terms": metadata.get("missing_terms", []),
        "ambiguities": metadata.get("ambiguities", []),
        "is_valid": precision["score"] >= 0.4,  # Medium or higher confidence
        "timestamp": datetime.now().isoformat()
    }
    
    return result

def print_result(result):
    """Print a query processing result in a readable format."""
    print("\n" + "=" * 80)
    print(f"Query: '{result['original_query']}'")
    
    if result["original_query"] != result["rewritten_query"]:
        print(f"Rewritten: '{result['rewritten_query']}'")
    
    print(f"Profile: {result['profile_name']} ({result['profile_id']})")
    print(f"Precision: {result['precision_score']:.2f} ({result['confidence_level'].upper()})")
    print(f"Valid for execution: {'YES' if result['is_valid'] else 'NO'}")
    print(f"Reasoning: {result['reasoning']}")
    
    # Term matches
    if result["column_matches"]:
        print("\nColumn Matches:")
        for term, matches in result["column_matches"].items():
            if matches:
                col = matches[0]["column"]
                conf = matches[0]["confidence"]
                print(f"  • '{term}' → '{col}' ({conf:.2f})")
    
    # Rewrites
    if result["rewrites"]:
        print("\nRewrites Applied:")
        for rewrite in result["rewrites"]:
            source = rewrite.get("source", "standard")
            print(f"  • '{rewrite['original']}' → '{rewrite['rewritten']}' ({source})")
    
    # Ambiguities
    if result["ambiguities"]:
        print("\nAmbiguous Terms:")
        for ambiguity in result["ambiguities"]:
            term = ambiguity["term"]
            candidates = [f"{c['column']} ({c['confidence']:.2f})" for c in ambiguity["candidates"]]
            print(f"  • '{term}' could be: {', '.join(candidates)}")
    
    # Missing terms
    if result["missing_terms"]:
        print("\nUnrecognized Terms:")
        print(f"  • {', '.join(result['missing_terms'])}")
    
    print("=" * 80)

def interactive_demo(components, processors):
    """Run an interactive demo."""
    print("\n===== WATCHDOG AI SEMANTIC LAYER DEMO =====")
    print("Enter queries to process, or commands:")
    print("  :profiles   - List available profiles")
    print("  :set NAME   - Switch to a different profile")
    print("  :quit       - Exit the demo")
    
    # Start with general manager profile
    current_profile = "general_manager"
    if current_profile not in processors:
        current_profile = list(processors.keys())[0]
    
    print(f"\nUsing profile: {current_profile}")
    
    while True:
        try:
            user_input = input("\n> ")
            
            if user_input.lower() in [":quit", ":exit", "quit", "exit"]:
                break
                
            elif user_input.lower() == ":profiles":
                print("Available profiles:")
                for pid in processors.keys():
                    name = processors[pid]["profile"].name
                    role = processors[pid]["profile"].role.value
                    print(f"  • {pid}: {name} ({role})")
                    
            elif user_input.lower().startswith(":set "):
                new_profile = user_input[5:].strip()
                if new_profile in processors:
                    current_profile = new_profile
                    print(f"Switched to profile: {current_profile}")
                else:
                    print(f"Profile not found: {new_profile}")
                    print(f"Available profiles: {', '.join(processors.keys())}")
                    
            elif user_input.strip():
                # Process as a query
                result = process_query(user_input, current_profile, processors)
                if result:
                    print_result(result)
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
    
    print("\nThank you for using the Watchdog AI Semantic Layer Demo!")
    
def main():
    """Main demo function."""
    print("\nWatchdog AI Semantic Layer & Business Guardrails Demo")
    print("====================================================")
    
    # Load components
    components = load_components()
    
    # Create processors
    processors = create_processors(components)
    
    # Run some test queries
    test_queries = [
        "What is our total gross profit?",
        "Show me front end gross by salesperson",
        "How many units did we sell last month?",
        "What's the average PVR for new vehicles?",
        "Compare front and back end gross",
        "What's our net profit for Q1?",
        "Show me the revenue by banana type"  # Nonsensical query for testing
    ]
    
    print("\n===== SAMPLE QUERIES =====")
    for query in test_queries:
        # Process with general manager profile
        result = process_query(query, "general_manager", processors)
        if result:
            print_result(result)
    
    # Run interactive demo
    interactive_demo(components, processors)

if __name__ == "__main__":
    main()