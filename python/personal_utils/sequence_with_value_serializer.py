import struct
from builtins import bytes
import array


def vector_to_pvector(v):
    return bytes([len(v)]) + bytes(v)


def read_vector_from_pvector(read_bytes, number_format='d'):
    length = read_bytes()[0]
    return array.array(number_format, read_bytes(length))


def string_to_pstring(s, encoding='ascii'):
    bstring = s.encode(encoding)
    return bytes([len(bstring)]) + bytes(bstring)


def read_string_from_pstring(read_bytes, encoding='ascii'):
    length = read_bytes()[0]
    return read_bytes(length).decode(encoding)


class Serializer():
    STRING_KEY_FORMAT = 's'

    def __init__(self, encoding='ascii', key_format=STRING_KEY_FORMAT):
        self.value_packer = struct.Struct('d')
        self.encoding = encoding
        self.key_format = key_format

    def save(self, key_value_map, file_name):
        if self.key_format == self.STRING_KEY_FORMAT:
            return self._save_string_value_map(key_value_map, file_name)
        else:
            return self._save_vector_value_map(key_value_map, file_name)

    def load(self, file_name):
        if self.key_format == self.STRING_KEY_FORMAT:
            return self._load_string_value_map(file_name)
        else:
            return self._load_vector_value_map(file_name)

    def _save_vector_value_map(self, vector_to_value_map, file_name):
        data = bytes()
        for sequence, value in vector_to_value_map.items():
            data += (
                vector_to_pvector(sequence) +
                self.value_packer.pack(value)
            )
        with open(file_name + ".swv", 'wb') as f:
            f.write(data)

    def _load_vector_value_map(self, file_name):
        vector_to_value_map = {}
        with open(file_name + ".swv", 'rb') as f:
            while True:
                try:
                    sequence = read_vector_from_pstring(
                        lambda n=1: f.read(n),
                        number_format=self.number_format
                    )
                    vector_to_value_map[
                        sequence
                    ] = self.value_packer.unpack(
                        f.read(self.value_packer.size)
                    )[0]
                except IndexError:
                    break
        return string_to_value_map

    def _save_string_value_map(self, string_to_value_map, file_name):
        data = bytes()
        for sequence, value in string_to_value_map.items():
            data += (
                string_to_pstring(sequence, self.encoding) +
                self.value_packer.pack(value)
            )
        with open(file_name + ".swv", 'wb') as f:
            f.write(data)

    def _load_string_value_map(self, file_name):
        string_to_value_map = {}
        with open(file_name + ".swv", 'rb') as f:
            while True:
                try:
                    sequence = read_string_from_pstring(
                        lambda n=1: f.read(n),
                        self.encoding
                    )
                    string_to_value_map[
                        sequence
                    ] = self.value_packer.unpack(
                        f.read(self.value_packer.size)
                    )[0]
                except IndexError:
                    break
        return string_to_value_map
