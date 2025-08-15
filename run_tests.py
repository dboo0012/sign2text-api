#!/usr/bin/env python3
"""
Test runner script for sign2text API
"""
import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔍 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def get_test_statistics():
    """Get basic test statistics without running tests"""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "-q"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            # Count collected tests from output
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "tests collected" in line:
                    # Extract number from line like "34 tests collected in 0.02s"
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        return int(parts[0])
        return 0
    except:
        return 0


def main():
    """Run all test suites automatically"""
    print("🚀 sign2text API Automated Test Runner")
    print("=" * 70)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print(f"📁 Working directory: {project_dir}")
    
    # Get test statistics
    total_test_count = get_test_statistics()
    print(f"📊 Total tests discovered: {total_test_count}")
    print()
    
    test_suites = []
    
    # Test Suite 1: Fast Unit Tests (WebSocket, Models, etc.)
    suite_1 = {
        "name": "Fast Unit Tests", 
        "cmd": ["python", "-m", "pytest", "-v", "-m", "not slow", "--tb=short"],
        "description": "WebSocket handlers, models, and mocked tests"
    }
    test_suites.append(suite_1)
    
    # Test Suite 2: MediaPipe Tests (Mocked)
    suite_2 = {
        "name": "MediaPipe Tests (Mocked)",
        "cmd": ["python", "-m", "pytest", "-v", "-m", "mediapipe and not slow", "--tb=short"],
        "description": "Pose processing with mocked MediaPipe"
    }
    test_suites.append(suite_2)
    
    # Test Suite 3: Integration Tests  
    suite_3 = {
        "name": "Integration Tests",
        "cmd": ["python", "-m", "pytest", "-v", "-m", "integration", "--tb=short"],
        "description": "Component interaction tests"
    }
    test_suites.append(suite_3)
    
    # Test Suite 4: All Tests (Comprehensive)
    suite_4 = {
        "name": "All Tests (Except Slow)",
        "cmd": ["python", "-m", "pytest", "-v", "-m", "not slow", "--tb=line"],
        "description": "Complete test suite excluding slow tests"
    }
    test_suites.append(suite_4)
    
    # Run all test suites
    results = []
    
    for i, suite in enumerate(test_suites, 1):
        print(f"\n{'='*70}")
        print(f"🧪 Test Suite {i}/{len(test_suites)}: {suite['name']}")
        print(f"📝 {suite['description']}")
        print(f"{'='*70}")
        
        success = run_command(suite['cmd'], suite['name'])
        results.append({
            'name': suite['name'],
            'success': success,
            'description': suite['description']
        })
    
    # Final Summary
    print("\n" + "=" * 70)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    print(f"📊 Test Suites: {passed}/{len(results)} passed")
    print(f"🎯 Total Tests: {total_test_count} discovered")
    
    # Detailed results
    for i, result in enumerate(results, 1):
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{i}. {result['name']:<25} {status}")
        print(f"   {result['description']}")
    
    print("\n" + "=" * 70)
    
    if passed == len(results):
        print("🎉 ALL TEST SUITES PASSED!")
        sys.exit(0)
    else:
        print(f"⚠️  {failed} TEST SUITE(S) FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
