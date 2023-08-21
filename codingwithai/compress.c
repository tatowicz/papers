#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <zlib.h>

void compress_data(const char *input, char *output, int input_length, uLongf *output_length) {
    compress((Bytef *)output, output_length, (const Bytef *)input, input_length);
}

void decompress_data(const char *input, char *output, int input_length, uLongf output_length) {
    uncompress((Bytef *)output, &output_length, (const Bytef *)input, input_length);
}

int main() {
    char data[] = "This is a sample binary data. This is a sample binary data.";
    uLongf compressed_size = compressBound(strlen(data));
    char compressed_data[compressed_size];
    char decompressed_data[sizeof(data)];

    compress_data(data, compressed_data, strlen(data), &compressed_size);
    decompress_data(compressed_data, decompressed_data, compressed_size, strlen(data));

    printf("Original data: %s\n", data);
    printf("Compressed data: %s\n", compressed_data);

    assert(strcmp(data, decompressed_data) == 0);
    printf("Data compressed and decompressed successfully!\n");
    return 0;
}