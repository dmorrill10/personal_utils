from personal_utils.sequence_with_value_serializer import *


def test_string_to_pstring():
    s = 'hi'
    patient = string_to_pstring(s)
    assert patient[0] == 2
    assert patient[1:].decode('ascii') == 'hi'


def test_pstring_to_string():
    s = 'hi'

    byte_list = [chr(2).encode('ascii')] + [c.encode('ascii') for c in s]
    byte_list.reverse()

    def read_bytes(n=1):
        bt = bytes()
        for _ in range(n):
            bt += byte_list.pop()
        return bt

    patient = read_string_from_pstring(
        read_bytes
    )
    assert patient == 'hi'


def test_creation():
    patient = Serializer()


def test_save():
    patient = Serializer()
    data = {'s1': 3.1, 's2': 1.234}
    # @todo Tempfile
    # patient.save(data)
