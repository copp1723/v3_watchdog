"""
Example script demonstrating the CSV/PDF Ingestion & Normalization Engine.

This example shows how to use the IngestionOrchestrator to process CSV and PDF files,
apply schema normalization, and generate validation reports.
"""

import os
import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.watchdog_ai.utils.ingestion_orchestrator import IngestionOrchestrator

def main():
    """Main entry point for the example script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CSV/PDF Ingestion & Normalization Example")
    parser.add_argument("--file", type=str, help="Path to a single file (CSV or PDF)")
    parser.add_argument("--dir", type=str, help="Path to a directory containing files to ingest")
    parser.add_argument("--pattern", type=str, default="*.csv", help="File pattern when using directory (e.g., '*.csv')")
    parser.add_argument("--dealer", type=str, help="Dealer ID for schema selection")
    parser.add_argument("--vendor", type=str, default="example", help="Vendor name for lineage tracking")
    parser.add_argument("--output", type=str, default="data/output", help="Output directory for processed data and reports")
    args = parser.parse_args()
    
    # Ensure at least one input option is provided
    if not args.file and not args.dir:
        parser.error("Either --file or --dir must be specified")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize the orchestrator
    print(f"Initializing IngestionOrchestrator...")
    orchestrator = IngestionOrchestrator(output_dir=args.output)
    
    # Process a single file
    if args.file:
        file_path = os.path.abspath(args.file)
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
        
        print(f"Processing file: {file_path}")
        result = orchestrator.ingest_file(
            file_path,
            dealer_id=args.dealer,
            vendor=args.vendor
        )
        
        if result.success:
            print(f"✅ Successfully processed file with {len(result.dataframe)} rows and {len(result.dataframe.columns)} columns")
            
            # Save the processed DataFrame
            output_csv = os.path.join(args.output, "processed_data.csv")
            result.dataframe.to_csv(output_csv, index=False)
            print(f"Processed data saved to: {output_csv}")
            
            # Print validation summary
            print("\nValidation Summary:")
            print(f"- File: {result.source_file}")
            print(f"- Processing time: {result.ingestion_time}")
            
            if "parsing_result" in result.validation_summary:
                parse_info = result.validation_summary["parsing_result"]
                if "file_type" in parse_info:
                    print(f"- File type: {parse_info['file_type']}")
            
            if "normalization_result" in result.validation_summary:
                norm_info = result.validation_summary["normalization_result"]
                if "mapped_column_count" in norm_info:
                    print(f"- Mapped columns: {norm_info['mapped_column_count']}")
                if "unmapped_columns" in norm_info:
                    if norm_info["unmapped_columns"]:
                        print(f"- Unmapped columns: {', '.join(norm_info['unmapped_columns'])}")
                    else:
                        print("- All columns mapped successfully")
        else:
            print(f"❌ Error processing file: {result.error_message}")
    
    # Process a directory of files
    elif args.dir:
        dir_path = os.path.abspath(args.dir)
        if not os.path.isdir(dir_path):
            print(f"Error: Directory not found: {dir_path}")
            return
        
        print(f"Processing files in directory: {dir_path} with pattern: {args.pattern}")
        combined_df, results = orchestrator.ingest_directory(
            dir_path,
            file_pattern=args.pattern,
            dealer_id=args.dealer,
            vendor=args.vendor,
            combine_results=True
        )
        
        success_count = sum(1 for r in results if r.success)
        failure_count = sum(1 for r in results if not r.success)
        
        print(f"\nProcessing Results:")
        print(f"- Total files: {len(results)}")
        print(f"- Successfully processed: {success_count}")
        print(f"- Failed: {failure_count}")
        
        if combined_df is not None:
            print(f"- Combined data: {len(combined_df)} rows and {len(combined_df.columns)} columns")
            
            # Save the combined DataFrame
            output_csv = os.path.join(args.output, "combined_data.csv")
            combined_df.to_csv(output_csv, index=False)
            print(f"Combined data saved to: {output_csv}")
            
            # Create a summary report for all files
            summary_report = {
                "timestamp": datetime.now().isoformat(),
                "directory": dir_path,
                "pattern": args.pattern,
                "total_files": len(results),
                "successful_files": success_count,
                "failed_files": failure_count,
                "file_results": [
                    {
                        "file": r.source_file,
                        "success": r.success,
                        "error": r.error_message if not r.success else None,
                        "row_count": len(r.dataframe) if r.success and r.dataframe is not None else 0
                    }
                    for r in results
                ]
            }
            
            # Save the summary report
            summary_path = os.path.join(args.output, "ingestion_summary_report.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_report, f, indent=2)
            print(f"Summary report saved to: {summary_path}")
        else:
            print("No data was successfully processed from any files")

if __name__ == "__main__":
    main()