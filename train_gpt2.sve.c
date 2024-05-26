#include <unistd.h>
#include <time.h>
#include <assert.h>
#include "dev/sve/gpt2_sve.h"
#include "dev/sve/utils+tokenizer.h"

// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

typedef struct {
    // hyperparameters
    int B; // batch size
    int T; // sequence length
    // input handling and its state
    FILE* tokens_file;
    long file_size;
    long current_position;
    // output memory
    int* batch;
    int* inputs;
    int* targets;
    // convenience variables
    int num_batches;
} DataLoader;

void dataloader_init(DataLoader *loader, const char* filename, int B, int T) {
    loader->B = B;
    loader->T = T;

    // open the input file for reading
    loader->tokens_file = fopen(filename, "rb");
    if (loader->tokens_file == NULL) {
        printf("Error opening tokens file\n");
        exit(1);
    }

    // determine the file size
    fseekCheck(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseekCheck(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(1);
    }
    loader->current_position = 0; // start at the beginning

    // allocate space for B*T + 1 integers to store the inputs and targets
    loader->batch = (int*) mallocCheck((B * T + 1) * sizeof(int));
    loader->inputs = loader->batch;
    loader->targets = loader->batch + 1; // targets are shifted by one
    loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) {
    loader->current_position = 0;
}

void dataloader_next_batch(DataLoader *loader) {
    int B = loader->B;
    int T = loader->T;
    // if we are at the end of the file, loop back to the beginning
    if (loader->current_position + (B*T+1) * sizeof(int) > loader->file_size) {
        loader->current_position = 0;
    }
    // read the B*T+1 integers from the file into batch
    fseekCheck(loader->tokens_file, loader->current_position, SEEK_SET);
    freadCheck(loader->batch, sizeof(int), B*T+1, loader->tokens_file);
    // advance the current position by B*T integers
    loader->current_position += B*T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
    fcloseCheck(loader->tokens_file);
    free(loader->batch);
}

// ----------------------------------------------------------------------------
// sampler

// the GPT-2 end-of-text token id
#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
int main() {
    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "data/TinyStories_train.bin";
    const char* tiny_stories_val = "data/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader;
    dataloader_init(&train_loader, train_tokens, B, T);
    printf("train dataset num_batches: %d\n", train_loader.num_batches);
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T);
    printf("val dataset num_batches: %d\n", val_loader.num_batches);
    int val_num_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64; // number of steps of inference we will do

    // train
    struct timespec start, end;
    for (int step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = GPT2_EOT;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                float* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}