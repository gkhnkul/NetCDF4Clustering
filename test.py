import assignment
import pytest

def test_addseconds():
    assert addSeconds("2018-01-01 00:21:01", 5) == "2018-01-01 00:21:06"