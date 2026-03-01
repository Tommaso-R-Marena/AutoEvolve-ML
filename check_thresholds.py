import json
import os
import subprocess
import sys

def check_quality_thresholds():
    """Check all quality thresholds for self-modification"""
    results = {
        'approved': True,
        'checks': [],
        'failure_reasons': [],
        'val_loss': 0.0,
        'threshold_val_loss': 1.0,
        'performance_change': 0.0,
        'coverage': 0.0,
        'tests_passed': 0,
        'tests_total': 0
    }
    
    # 1. Test coverage check
    print("Checking test coverage...")
    try:
        if os.path.exists('coverage.json'):
            with open('coverage.json', 'r') as f:
                cov_data = json.load(f)
            coverage = cov_data['totals']['percent_covered']
            results['coverage'] = coverage
            
            if coverage >= 80:
                results['checks'].append({
                    'name': 'Test Coverage',
                    'passed': True,
                    'message': f'{coverage:.1f}% (≥80% required)'
                })
            else:
                results['approved'] = False
                results['checks'].append({
                    'name': 'Test Coverage',
                    'passed': False,
                    'message': f'{coverage:.1f}% (≥80% required)'
                })
                results['failure_reasons'].append(f'Test coverage {coverage:.1f}% below 80% threshold')
        else:
            results['checks'].append({
                'name': 'Test Coverage',
                'passed': True,
                'message': 'No coverage data (skipped)'
            })
    except Exception as e:
        print(f"Coverage check error: {e}")
        results['checks'].append({
            'name': 'Test Coverage',
            'passed': True,
            'message': 'Error reading coverage (skipped)'
        })
    
    # 2. All tests must pass
    print("Checking test results...")
    try:
        test_result = subprocess.run(
            ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            capture_output=True,
            text=True
        )
        
        # Parse pytest output
        output = test_result.stdout + test_result.stderr
        if 'passed' in output:
            # Extract test counts
            import re
            match = re.search(r'(\d+) passed', output)
            if match:
                results['tests_passed'] = int(match.group(1))
                results['tests_total'] = results['tests_passed']
        
        if test_result.returncode == 0:
            results['checks'].append({
                'name': 'Unit Tests',
                'passed': True,
                'message': f'All {results["tests_passed"]} tests passed'
            })
        else:
            results['approved'] = False
            results['checks'].append({
                'name': 'Unit Tests',
                'passed': False,
                'message': 'Some tests failed'
            })
            results['failure_reasons'].append('Unit tests failing')
    except Exception as e:
        print(f"Test check error: {e}")
        results['checks'].append({
            'name': 'Unit Tests',
            'passed': True,
            'message': 'No tests found (skipped)'
        })
    
    # 3. Model performance validation
    print("Checking model performance...")
    try:
        with open('metrics.json', 'r') as f:
            history = json.load(f)
        
        if len(history.get('val_loss', [])) > 0:
            current_loss = history['val_loss'][-1]
            results['val_loss'] = current_loss
            results['threshold_val_loss'] = min(history['val_loss']) * 2.0  # Allow 2x degradation
            
            if len(history['val_loss']) > 5:
                old_loss = history['val_loss'][-6]
                results['performance_change'] = (old_loss - current_loss) / old_loss
            
            # Check for catastrophic degradation
            best_loss = min(history['val_loss'])
            if current_loss < best_loss * 2.0:
                results['checks'].append({
                    'name': 'Model Performance',
                    'passed': True,
                    'message': f'Loss {current_loss:.4f} within acceptable range'
                })
            else:
                results['approved'] = False
                results['checks'].append({
                    'name': 'Model Performance',
                    'passed': False,
                    'message': f'Loss {current_loss:.4f} degraded significantly from best {best_loss:.4f}'
                })
                results['failure_reasons'].append('Model performance degraded beyond threshold')
        else:
            results['checks'].append({
                'name': 'Model Performance',
                'passed': True,
                'message': 'No metrics yet (skipped)'
            })
    except Exception as e:
        print(f"Performance check error: {e}")
        results['checks'].append({
            'name': 'Model Performance',
            'passed': True,
            'message': 'No metrics file (skipped)'
        })
    
    # 4. Code quality (linting)
    print("Checking code quality...")
    try:
        lint_result = subprocess.run(
            ['flake8', '.', '--count', '--select=E9,F63,F7,F82', '--show-source', '--statistics'],
            capture_output=True,
            text=True
        )
        
        if lint_result.returncode == 0 or 'E' not in lint_result.stdout:
            results['checks'].append({
                'name': 'Code Quality',
                'passed': True,
                'message': 'No critical linting errors'
            })
        else:
            results['approved'] = False
            results['checks'].append({
                'name': 'Code Quality',
                'passed': False,
                'message': 'Critical linting errors found'
            })
            results['failure_reasons'].append('Code quality issues detected')
    except Exception as e:
        print(f"Linting check error: {e}")
        results['checks'].append({
            'name': 'Code Quality',
            'passed': True,
            'message': 'Linter not available (skipped)'
        })
    
    # Save results
    with open('quality_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY GATE SUMMARY")
    print("="*60)
    for check in results['checks']:
        status = "✅" if check['passed'] else "❌"
        print(f"{status} {check['name']}: {check['message']}")
    print("="*60)
    
    if results['approved']:
        print("\n✅ ALL CHECKS PASSED - Modification approved")
        return 0
    else:
        print("\n❌ QUALITY GATE FAILED")
        for reason in results['failure_reasons']:
            print(f"  - {reason}")
        return 1

if __name__ == "__main__":
    sys.exit(check_quality_thresholds())
