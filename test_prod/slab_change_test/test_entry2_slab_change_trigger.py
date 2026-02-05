#!/usr/bin/env python3
"""
DEPRECATED (kept for backwards compatibility).

Use `test_prod/slab_change_test/test_entry2_slab_change.py` instead.
This wrapper runs the consolidated core Entry2 trigger detection test.
"""

def test_entry2_trigger_detection():
    # Import consolidated test module as a sibling (test_prod is not a Python package by default).
    from test_entry2_slab_change import test_entry2_trigger_detection_basic
    test_entry2_trigger_detection_basic()

if __name__ == "__main__":
    test_entry2_trigger_detection()
    print("OK")

