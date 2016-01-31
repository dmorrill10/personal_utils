import struct


def string_to_pstring(s, encoding='ascii'):
    bstring = s.encode(encoding)
    return chr(len(bstring)).encode(encoding) + bstring


def read_string_from_pstring(read_bytes, encoding='ascii'):
    length = read_bytes()[0]
    return read_bytes(length).decode(encoding)


class Serializer():
    def __init__(self, encoding='ascii'):
        self.value_packer = struct.Struct('d')
        self.encoding = encoding

    def save(self, sequences_with_values, file_name):
        data = bytes()
        for sequence, value in sequences_with_values.items():
            data += (
                string_to_pstring(sequence, self.encoding) +
                self.value_packer.pack(value)
            )
        with open(file_name + ".swv", 'wb') as f:
            f.write(data)

    def load(self, file_name):
        sequences_with_values = {}
        with open(file_name + ".swv", 'rb') as f:
            while True:
                try:
                    sequence = read_string_from_pstring(
                        lambda n=1: f.read(n),
                        self.encoding
                    )
                    sequences_with_values[
                        sequence
                    ] = self.value_packer.unpack(
                        f.read(self.value_packer.size)
                    )[0]
                except IndexError:
                    break
        return sequences_with_values
