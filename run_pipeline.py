"""
Master Script to Run the Entire Machine Learning Pipeline
This script executes all steps of the ML project in the correct order.
"""

import subprocess
import sys
import os

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}:")
        print(e.stderr)
        return False

def main():
    """Execute the complete ML pipeline."""
    print("=" * 80)
    print("COLLEGE STUDENT PLACEMENT PREDICTION - ML PIPELINE")
    print("=" * 80)
    print("\nThis will run all scripts in the correct order:")
    print("1. Data Exploration")
    print("2. Data Preprocessing & Visualization")
    print("3. Model Training & Evaluation")
    print("4. Report Generation")
    print("5. Final Summary")
    print("\n" + "=" * 80)
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Define scripts in execution order
    scripts = [
        ("src/explore_data.py", "Step 1: Data Exploration"),
        ("src/visualize_and_preprocess.py", "Step 2: Preprocessing & Visualization"),
        ("src/train_models.py", "Step 3: Model Training & Evaluation"),
        ("src/generate_report.py", "Step 4: Report Generation"),
        ("src/create_final_summary.py", "Step 5: Final Summary")
    ]
    
    # Execute each script
    results = []
    for script_path, description in scripts:
        success = run_script(script_path, description)
        results.append((description, success))
        
        if not success:
            print(f"\n⚠ Warning: {description} failed. Continuing with next steps...")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    all_success = all(success for _, success in results)
    
    print("\n" + "=" * 80)
    if all_success:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n✅ Project is ready for submission!")
        print("\n📁 Check the following folders:")
        print("  • data/ - All data files")
        print("  • visualizations/ - All generated charts")
        print("  • reports/ - Analysis report")
        print("  • models/ - Saved models (if any)")
    else:
        print("⚠ PIPELINE COMPLETED WITH SOME ERRORS")
        print("=" * 80)
        print("\nPlease check the error messages above and re-run failed steps.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
