#pragma once

#include <stdio.h>
#include <stdint.h>
#include <ctype.h>
#include <assert.h>

FILE *fopen_check(const char *path, const char *mode, const char *file, int line);
#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line);
#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

void fclose_check(FILE *fp, const char *file, int line);
#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

void fseek_check(FILE *fp, long off, int whence, const char *file, int line);
#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

void *malloc_check(size_t size, const char *file, int line);
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

///  TOKENIZER

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> token id
} Tokenizer;

void tokenizer_init(Tokenizer *tokenizer, const char *filename);
const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id);
void tokenizer_free(Tokenizer *tokenizer);
void safe_printf(const char *piece);