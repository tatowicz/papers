import zlib

def compress(data: bytes) -> bytes:
    return zlib.compress(data)

def decompress(data: bytes) -> bytes:
    return zlib.decompress(data)

# Example usage:
data = b"Your binary data here"
compressed_data = compress(data)
decompressed_data = decompress(compressed_data)
print(compressed_data)
print(decompressed_data)
assert data == decompressed_data