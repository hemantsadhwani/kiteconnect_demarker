#!/usr/bin/env python3
"""
DEPRECATED (kept for backwards compatibility).

Use `test_prod/slab_change_test/test_entry2_slab_change.py` instead.
This wrapper runs the consolidated slab-change Entry2 handoff test.
"""


def test_entry2_handoff_sets_new_symbol_state():
    # Import consolidated test module as a sibling (test_prod is not a Python package by default).
    from test_entry2_slab_change import test_entry2_slab_change_handoff_to_new_symbol
    test_entry2_slab_change_handoff_to_new_symbol()


if __name__ == "__main__":
    test_entry2_handoff_sets_new_symbol_state()
    print("OK")


