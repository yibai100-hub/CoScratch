"""Repeatable local verification. Never imports Arm_Lib, ROS or Isaac."""
from pathlib import Path
import json
import platform
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    begin=time.time()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    import torch,numpy,scipy
    report=dict(status='PASS'if result.wasSuccessful()else'FAIL',tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),elapsed_s=time.time()-begin,
                python=sys.version,platform=platform.platform(),torch=torch.__version__,
                numpy=numpy.__version__,scipy=scipy.__version__,hardware_commands_sent=False,
                physical_validation_passed=False,actual_network_scope='loopback two-process mock gateway')
    (ROOT/'reports/pc_validation.json').write_text(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful()else 1)
