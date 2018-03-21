import assignment
import pytest

# this method is a unit test for addseconds method
def test_addseconds():
    assert addSeconds(datetime.strptime("2018-01-01 00:21:01", '%Y-%m-%d %H:%M:%S'), 5) == datetime.strptime("2018-01-01 00:21:06", '%Y-%m-%d %H:%M:%S')

# this method is a unit test for readCDFData()
def test_readCDFData():
	assert readCDFData() == True