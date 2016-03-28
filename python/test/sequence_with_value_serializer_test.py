from personal_utils.sequence_with_value_serializer import *
import array


def test_vector_to_pvector():
    s = array.array('I', [0, 1, 0, 0, 1, 0])
    patient = vector_to_pvector(s)
    assert patient[0] == len(s)
    assert array.array('I', patient[1:]) == s


def test_pvector_to_vector():
    s = array.array('I', [0, 1, 0, 0, 1, 0])

    byte_list = list(bytes(chr(len(s)).encode('ascii')) + bytes(s))
    byte_list.reverse()

    def read_bytes(n=1):
        bt = bytes()
        for _ in range(n):
            bt += byte_list.pop()
        return bt

    patient = read_vector_from_pvector(
        read_bytes, number_format='I'
    )
    assert patient == s


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


def test_save_string_value_map():
    patient = Serializer()
    data = {'s1': 3.1, 's2': 1.234}
    # @todo Tempfile
    # patient.save(data)
