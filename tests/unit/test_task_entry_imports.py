# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual literal script startup without site preloading stdlib types."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MODULES = Path(__file__).resolve().parents[2] / 'pypto_serving/serving/native_ddr'


class StandaloneEntryTests(unittest.TestCase):
    def run_entry(self, name, code, marker):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', '-S', str(MODULES / name), '--help'],
                cwd=directory, capture_output=True, text=True, timeout=5)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, code, output)
        self.assertIn(marker, output)
        self.assertNotIn('partially initialized module', output)

    def test_supervisor_reaches_its_real_argument_parser(self):
        self.run_entry('task_supervisor.py', 0, '--config')

    def test_static_bootstrap_reaches_its_real_entry_guard(self):
        self.run_entry('task_lifecycle.py', 1, 'static task bootstrap requires a literal Python file')

    def test_reaper_reaches_its_real_config_read(self):
        self.run_entry('task_resources.py', 1, 'FileNotFoundError')


if __name__ == '__main__':
    unittest.main()
