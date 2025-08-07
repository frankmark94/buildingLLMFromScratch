#!/usr/bin/env python3
"""
Test runner for neural LLM project.
Runs all unit tests and generates coverage report.
"""

import sys
import unittest
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def discover_tests(test_dir: str = None, pattern: str = 'test_*.py') -> unittest.TestSuite:
    """Discover and load all test cases."""
    if test_dir is None:
        test_dir = str(Path(__file__).parent)
    
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    
    return suite


def run_tests(verbosity: int = 2, failfast: bool = False) -> unittest.TestResult:
    """Run all tests and return results."""
    suite = discover_tests()
    
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        stream=sys.stdout
    )
    
    result = runner.run(suite)
    return result


def print_test_summary(result: unittest.TestResult) -> None:
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILED TESTS:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERROR TESTS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
    
    print("="*60)


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Run neural LLM tests")
    
    parser.add_argument("--verbose", "-v", action="count", default=2,
                       help="Increase test output verbosity")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal test output")
    parser.add_argument("--failfast", "-f", action="store_true",
                       help="Stop on first failure")
    parser.add_argument("--pattern", "-p", default="test_*.py",
                       help="Test file pattern (default: test_*.py)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available tests")
    
    args = parser.parse_args()
    
    # Adjust verbosity
    if args.quiet:
        verbosity = 0
    else:
        verbosity = args.verbose
    
    print("🧪 Neural LLM Test Suite")
    print("="*40)
    
    if args.list:
        # List all tests
        suite = discover_tests(pattern=args.pattern)
        print(f"Found {suite.countTestCases()} test cases:")
        
        def extract_tests(test_suite):
            tests = []
            for test in test_suite:
                if isinstance(test, unittest.TestSuite):
                    tests.extend(extract_tests(test))
                else:
                    tests.append(test)
            return tests
        
        all_tests = extract_tests(suite)
        for test in sorted(all_tests, key=lambda x: str(x)):
            print(f"  - {test}")
        
        return 0
    
    # Run tests
    try:
        result = run_tests(verbosity=verbosity, failfast=args.failfast)
        
        # Print summary
        print_test_summary(result)
        
        # Return appropriate exit code
        return 0 if result.wasSuccessful() else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())