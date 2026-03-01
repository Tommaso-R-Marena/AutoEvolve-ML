import sys
import json
import os
from datetime import datetime
sys.path.insert(0, '.')

try:
    from research_automation import ResearchAutomation
    
    ra = ResearchAutomation()
    ra.load_research_state()
    
    report = ra.generate_research_report()
    
    with open('weekly_research_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    with open('report_metrics.txt', 'w') as f:
        f.write(f"BREAKTHROUGHS={report.get('breakthroughs', 0)}\n")
        f.write(f"EXPERIMENTS={report.get('total_experiments', 0)}\n")
    
    print('Research report generated')
    print(f"Total experiments: {report.get('total_experiments', 0)}")
    print(f"Breakthroughs: {report.get('breakthroughs', 0)}")
except Exception as e:
    print(f'Research report skipped: {e}')
    with open('report_metrics.txt', 'w') as f:
        f.write("BREAKTHROUGHS=0\n")
        f.write("EXPERIMENTS=0\n")
