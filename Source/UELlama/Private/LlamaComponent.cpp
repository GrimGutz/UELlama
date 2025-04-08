#include "UELlama/LlamaComponent.h"

using namespace std;

namespace {

    // === SAFE TOKENIZER WITH LOGGING ===
    vector<llama_token> my_llama_tokenize(llama_model* model, const string& text, vector<llama_token>& out, bool add_bos) {
        const llama_vocab* vocab = llama_model_get_vocab(model);
        out.resize(text.length() + 8);

        bool add_special = add_bos && llama_vocab_get_add_bos(vocab);
        bool parse_special = false;

        UE_LOG(LogTemp, Warning, TEXT("Tokenizing text: %s | Add BOS: %d"), *FString(UTF8_TO_TCHAR(text.c_str())), add_bos);

        int n = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int>(text.length()),
            out.data(),
            static_cast<int>(out.size()),
            add_special,
            parse_special
        );

        if (n < 0) {
            UE_LOG(LogTemp, Error, TEXT("Tokenization failed: %s"), *FString(UTF8_TO_TCHAR(text.c_str())));
            out.clear();
        }
        else {
            out.resize(n);
            FString tokensLog;
            for (int i = 0; i < n; ++i) {
                tokensLog += FString::Printf(TEXT("[%d]=%d "), i, out[i]);
            }
            UE_LOG(LogTemp, Warning, TEXT("Tokenized %d tokens: %s"), n, *tokensLog);
        }

        return out;
    }

    // === SAFE DETOKENIZER WITH LOGGING ===
    string llama_detokenize_bpe(llama_model* model, const vector<llama_token>& tokens) {
        const llama_vocab* vocab = llama_model_get_vocab(model);
        string result;
        vector<char> buffer(32);

        for (auto token : tokens) {
            int len = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, false);
            if (len < 0) {
                buffer.resize(-len);
                len = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, false);
            }
            if (len > 0) {
                result.append(buffer.data(), len);
            }
        }

        UE_LOG(LogTemp, Warning, TEXT("Detokenized string: %s"), *FString(UTF8_TO_TCHAR(result.c_str())));
        return result;
    }

}

namespace Internal {

    Llama::Llama() {
        UE_LOG(LogTemp, Warning, TEXT("Llama instance created (no thread yet)."));
    }

    Llama::~Llama() {
        running = false;
        if (thread.joinable()) {
            thread.join();
        }
    }

    void Llama::insertPrompt(FString v) {
        qMainToThread.enqueue([this, v = move(v)]() mutable {
            UE_LOG(LogTemp, Warning, TEXT("InsertPrompt called: %s"), *v);
            unsafeInsertPrompt(move(v));
            });
    }

    void Llama::unsafeInsertPrompt(FString v) {
        if (!ctx) {
            UE_LOG(LogTemp, Error, TEXT("Llama context not active."));
            return;
        }

        string stdV = " " + string(TCHAR_TO_UTF8(*v));
        vector<llama_token> line_inp;
        my_llama_tokenize(model, stdV, line_inp, false);

        UE_LOG(LogTemp, Warning, TEXT("Inserting prompt: %s | %d tokens"), *v, line_inp.size());

        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
    }

    void Llama::activate(bool bReset, Params params) {
        UE_LOG(LogTemp, Warning, TEXT("Llama activate called. Reset=%d"), bReset);

        // First run activation (this sets up the model + context)
        unsafeActivate(bReset, std::move(params));

        // Then start the thread
        if (!running) {
            running = true;
            thread = std::thread([this]() {
                UE_LOG(LogTemp, Warning, TEXT("Llama thread started from activate()"));
                threadRun();
                });
        }
    }


    void Llama::deactivate() {
        qMainToThread.enqueue([this]() {
            UE_LOG(LogTemp, Warning, TEXT("Llama deactivate called."));
            unsafeDeactivate();
            });
    }

    void Llama::unsafeActivate(bool bReset, Params params) {
        if (bReset) {
            UE_LOG(LogTemp, Warning, TEXT("Resetting Llama context..."));
            unsafeDeactivate();
        }

        if (model) {
            UE_LOG(LogTemp, Warning, TEXT("Model already initialized. Skipping reload."));
            return;
        }

        UE_LOG(LogTemp, Warning, TEXT("Initializing llama backend..."));
        llama_backend_init();

        llama_model_params modelParams = llama_model_default_params();
        UE_LOG(LogTemp, Warning, TEXT("Loading model from path: %s"), *params.pathToModel);

        model = llama_model_load_from_file(TCHAR_TO_UTF8(*params.pathToModel), modelParams);
        if (!model) {
            UE_LOG(LogTemp, Error, TEXT("Model load failed: %s"), *params.pathToModel);
            unsafeDeactivate();
            return;
        }

        UE_LOG(LogTemp, Warning, TEXT("Model loaded: %p"), model);

        llama_context_params ctxParams = llama_context_default_params();
        ctxParams.n_ctx = 2048;
        ctxParams.n_batch = 512;
        ctxParams.n_ubatch = 512;
        ctxParams.n_seq_max = 1;
        ctxParams.n_threads = n_threads;
        ctxParams.n_threads_batch = n_threads;

        ctx = llama_init_from_model(model, ctxParams);
        if (!ctx) {
            UE_LOG(LogTemp, Error, TEXT("Context initialization failed"));
            unsafeDeactivate();
            return;
        }

        // === WARMUP ONLY ===
        UE_LOG(LogTemp, Warning, TEXT("Running warmup decode with BOS token..."));
        llama_token bos_token = llama_vocab_bos(llama_model_get_vocab(model));
        llama_batch batch = llama_batch_init(1, 0, 1);
        if (!batch.token) {
            UE_LOG(LogTemp, Error, TEXT("Batch init failed"));
            llama_batch_free(batch);
            unsafeDeactivate();
            return;
        }

        batch.n_tokens = 1;
        batch.token[0] = bos_token;
        batch.pos[0] = 0;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        int result = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (result != 0) {
            UE_LOG(LogTemp, Error, TEXT("Warmup decode failed: %d"), result);
            unsafeDeactivate();
            return;
        }

        UE_LOG(LogTemp, Warning, TEXT("Warmup completed. Ready for prompts."));

        // Don't insert prompt or stop sequences yet — let the user do it later
        last_n_tokens.resize(ctxParams.n_ctx, 0);
        n_consumed = 0;
        n_past = 0;
        eos = false;
        embd_inp.clear(); // make sure

        UE_LOG(LogTemp, Warning, TEXT("LLaMA initialized. Waiting for input."));
    }


    void Llama::unsafeDeactivate() {
        UE_LOG(LogTemp, Warning, TEXT("🧼 Deactivating LLaMA..."));

        running = false;

        // Only join if we're not inside the thread itself
        if (std::this_thread::get_id() != thread.get_id()) {
            if (thread.joinable()) {
                UE_LOG(LogTemp, Warning, TEXT("🧵 Joining LLaMA thread..."));
                thread.join();
                UE_LOG(LogTemp, Warning, TEXT("✅ Thread joined."));
            }
        }
        else {
            UE_LOG(LogTemp, Warning, TEXT("⚠️ Skipped thread.join() — already inside thread."));
        }

        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }

        if (model) {
            llama_model_free(model);
            model = nullptr;
        }

        UE_LOG(LogTemp, Warning, TEXT("🔻 Deactivation complete."));
    }

    void Llama::threadRun() {
        UE_LOG(LogTemp, Warning, TEXT("%p 🔁 LLaMA thread started."), this);

        const int n_batch = 512;
        const int n_keep = 0;
        const float temperature = 0.8f;

        while (running) {
            // Process cross-thread messages
            while (qMainToThread.processQ()) {
                UE_LOG(LogTemp, Verbose, TEXT("✅ Processed main->thread message"));
            }

            // Wait until model and context are ready
            if (!ctx || !model) {
                if (!running) break;
                UE_LOG(LogTemp, Warning, TEXT("⏳ Context or model not ready."));
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            // If we hit EOS and nothing more to process
            if (eos && n_consumed >= embd_inp.size()) {
                UE_LOG(LogTemp, Warning, TEXT("🛑 EOS reached. Waiting for new prompt..."));
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            // Nothing to do
            if (embd_inp.empty() || n_consumed >= embd_inp.size()) {
                UE_LOG(LogTemp, Warning, TEXT("⏳ No input tokens to process. embd_inp.size()=%d, n_consumed=%d"),
                    static_cast<int>(embd_inp.size()), n_consumed);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            eos = false;
            const int n_ctx = llama_n_ctx(ctx);

            // Prevent context overflow: sliding window
            if (n_past + (int)embd.size() > n_ctx) {
                int n_left = n_past - n_keep;
                n_past = std::max(1, n_keep);

                embd.insert(
                    embd.begin(),
                    last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(),
                    last_n_tokens.end() - embd.size()
                );

                UE_LOG(LogTemp, Warning, TEXT("🔁 Context full — sliding window applied."));
            }

            // Run decode if needed
            if (!embd.empty()) {
                llama_batch batch = llama_batch_get_one(embd.data(), embd.size());
                int res = llama_decode(ctx, batch);
                llama_batch_free(batch);

                if (res != 0) {
                    UE_LOG(LogTemp, Error, TEXT("❌ llama_decode failed with code %d"), res);
                    continue;
                }

                n_past += embd.size();
            }

            embd.clear();

            // Consume prompt tokens
            while (n_consumed < embd_inp.size()) {
                llama_token tok = embd_inp[n_consumed++];
                embd.push_back(tok);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(tok);

                if ((int)embd.size() >= n_batch) break;
            }

            UE_LOG(LogTemp, Warning, TEXT("📥 Consumed user input tokens: %d"), embd.size());

            // Run one decode step if embd now has content
            if (!embd.empty()) {
                llama_batch batch = llama_batch_get_one(embd.data(), embd.size());
                int res = llama_decode(ctx, batch);
                llama_batch_free(batch);

                if (res != 0) {
                    UE_LOG(LogTemp, Error, TEXT("❌ llama_decode() failed again with %d"), res);
                    continue;
                }

                n_past += embd.size();

                // Generate next token (basic max_logit sampling for now)
                float* logits = llama_get_logits(ctx);
                if (!logits) {
                    UE_LOG(LogTemp, Error, TEXT("❌ Logits null."));
                    continue;
                }

                int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
                llama_token sampled = 0;
                float max_logit = logits[0];

                for (int i = 1; i < n_vocab; ++i) {
                    if (logits[i] > max_logit) {
                        max_logit = logits[i];
                        sampled = i;
                    }
                }

                embd.clear();
                embd.push_back(sampled);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(sampled);

                UE_LOG(LogTemp, Warning, TEXT("🧠 Sampled token: %d"), sampled);

                // Send detokenized result to main
                FString outStr = UTF8_TO_TCHAR(llama_detokenize_bpe(model, embd).c_str());
                qThreadToMain.enqueue([this, outStr = std::move(outStr)] {
                    if (tokenCb) tokenCb(outStr);
                    });

                if (sampled == llama_vocab_eos(llama_model_get_vocab(model))) {
                    eos = true;
                    UE_LOG(LogTemp, Warning, TEXT("🛑 EOS token detected. Ending inference."));
                }
            }
            if (!running) break;
        }

        UE_LOG(LogTemp, Warning, TEXT("👋 LLaMA thread shutting down."));
        unsafeDeactivate();
    }


    void Llama::process() {
        while (qThreadToMain.processQ()) {}
    }

} // namespace Internal

// === UELlamaComponent ===

ULlamaComponent::ULlamaComponent(const FObjectInitializer& ObjectInitializer)
    : UActorComponent(ObjectInitializer), llama(make_unique<Internal::Llama>()) {
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;

    llama->tokenCb = [this](FString NewToken) {
        OnNewTokenGenerated.Broadcast(move(NewToken));
        };
}

ULlamaComponent::~ULlamaComponent() = default;

void ULlamaComponent::Activate(bool bReset) {
    Super::Activate(bReset);
    Params params;
    params.prompt = prompt;
    params.pathToModel = pathToModel;
    params.stopSequences = stopSequences;
    llama->activate(bReset, move(params));
}

void ULlamaComponent::Deactivate() {
    llama->deactivate();
    Super::Deactivate();
}

void ULlamaComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) {
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    llama->process();
}

void ULlamaComponent::InsertPrompt(const FString& v) {
    llama->insertPrompt(v);
}
