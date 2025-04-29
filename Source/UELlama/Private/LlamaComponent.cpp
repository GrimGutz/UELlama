#include "UELlama/LlamaComponent.h"

using namespace std;

// --- LlamaInternal implementations ---

namespace LlamaInternal {

    void Q::enqueue(function<void()> func) {
        lock_guard<mutex> lock(mutex_);
        q.emplace_back(move(func));
    }

    bool Q::processQ() {
        function<void()> func;
        {
            lock_guard<mutex> lock(mutex_);
            if (q.empty()) return false;
            func = move(q.front());
            q.pop_front();
        }
        func();
        return true;
    }

    vector<llama_token> my_llama_tokenize(llama_model* model, const string& text, vector<llama_token>& out, bool add_bos) {
        const llama_vocab* vocab = llama_model_get_vocab(model);
        out.resize(text.length() + 8);

        bool add_special = add_bos && llama_vocab_get_add_bos(vocab);
        bool parse_special = false;

        UE_LOG(LogTemp, Warning, TEXT("Tokenizing text: %s | Add BOS: %d"), *FString(UTF8_TO_TCHAR(text.c_str())), add_bos);

        int n = llama_tokenize(vocab, text.c_str(), static_cast<int>(text.length()), out.data(), static_cast<int>(out.size()), add_special, parse_special);

        if (n < 0) {
            UE_LOG(LogTemp, Error, TEXT("Tokenization failed: %s"), *FString(UTF8_TO_TCHAR(text.c_str())));
            out.clear();
        }
        else {
            out.resize(n);
        }
        return out;
    }

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
        return result;
    }

} // namespace LlamaInternal

// --- Internal::Llama implementation ---

namespace Internal {

    Llama::Llama() {
        UE_LOG(LogTemp, Warning, TEXT("Llama instance created."));
    }

    Llama::~Llama() {
        running = false;
        if (thread.joinable()) {
            thread.join();
        }
    }

    void Llama::insertPrompt(FString userPrompt) {
        qMainToThread.enqueue([this, userPrompt = std::move(userPrompt)]() mutable {
            unsafeInsertPrompt(std::move(userPrompt));
            });
    }

    void Llama::unsafeInsertPrompt(FString userPrompt) {
        if (!ctx || !model) {
            UE_LOG(LogTemp, Error, TEXT("❌ Context not ready"));
            return;
        }

        // Add new user message to history
        chatHistory.emplace_back("user", TCHAR_TO_UTF8(*userPrompt));

        // Rebuild full prompt
        string fullPrompt = "### System:\n" + string(TCHAR_TO_UTF8(*systemPrompt)) + "\n\n";
        for (const auto& [role, msg] : chatHistory) {
            fullPrompt += "### " + role + ":\n" + msg + "\n\n";
        }
        fullPrompt += "### Assistant:\n";

        vector<llama_token> tokens;
        LlamaInternal::my_llama_tokenize(model, fullPrompt, tokens, true);

        if (tokens.empty()) {
            UE_LOG(LogTemp, Error, TEXT("⚠️ Failed to tokenize full chat prompt."));
            return;
        }

        embd_inp = move(tokens);
        n_consumed = 0;
        eos = false;
        assistant_ss.str(""); // clear assistant accumulation
    }

    void Llama::activate(bool bReset, LlamaInternal::Params params) {
        qMainToThread.enqueue([this, params = std::move(params), bReset]() mutable {
            unsafeActivate(bReset, std::move(params));
            });

        if (!running) {
            running = true;
            thread = std::thread([this]() { threadRun(); });
        }
    }

    void Llama::deactivate() {
        qMainToThread.enqueue([this]() { unsafeDeactivate(); });
    }

    void Llama::unsafeActivate(bool bReset, LlamaInternal::Params params) {
        if (bReset) {
            unsafeDeactivate();
        }

        if (model) {
            UE_LOG(LogTemp, Warning, TEXT("Model already loaded."));
            return;
        }

        llama_backend_init();

        llama_model_params modelParams = llama_model_default_params();
        model = llama_model_load_from_file(TCHAR_TO_UTF8(*params.pathToModel), modelParams);
        if (!model) {
            UE_LOG(LogTemp, Error, TEXT("❌ Failed to load model."));
            return;
        }

        llama_context_params ctxParams = llama_context_default_params();
        ctxParams.n_ctx = 2048;
        ctxParams.n_threads = LlamaInternal::n_threads;
        ctx = llama_init_from_model(model, ctxParams);
        if (!ctx) {
            UE_LOG(LogTemp, Error, TEXT("❌ Failed to initialize context."));
            llama_model_free(model);
            model = nullptr;
            return;
        }
        stopSequences.clear();

        for (const FString& seq : params.stopSequences) {
            std::string utf8_seq = TCHAR_TO_UTF8(*seq);
            std::vector<llama_token> seq_tokens;
            LlamaInternal::my_llama_tokenize(model, utf8_seq, seq_tokens, false);
            if (!seq_tokens.empty()) {
                stopSequences.push_back(std::move(seq_tokens));
                UE_LOG(LogTemp, Warning, TEXT("Registered stop sequence: %s (%d tokens)"), *seq, seq_tokens.size());
            }
        }

        systemPrompt = params.prompt;
        chatHistory.clear();
        assistant_ss.str("");
        embd_inp.clear();
        n_consumed = 0;
        n_past = 0;
        eos = false;
    }

    void Llama::unsafeDeactivate() {
        running = false;

        if (thread.joinable() && std::this_thread::get_id() != thread.get_id()) {
            thread.join();
        }

        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
    }

    void Llama::threadRun() {
        UE_LOG(LogTemp, Warning, TEXT("🔁 LLaMA thread started."));

        while (running) {
            while (qMainToThread.processQ()) {}

            if (!ctx || !model) {
                FPlatformProcess::Sleep(0.1f);
                continue;
            }

            if (eos) {
                FPlatformProcess::Sleep(0.1f);
                continue;
            }

            if (n_consumed >= embd_inp.size()) {
                FPlatformProcess::Sleep(0.1f);
                continue;
            }

            embd.clear();
            while (n_consumed < embd_inp.size() && embd.size() < 512) {
                embd.push_back(embd_inp[n_consumed++]);
            }

            llama_batch batch = llama_batch_get_one(embd.data(), embd.size());
            if (llama_decode(ctx, batch) != 0) {
                UE_LOG(LogTemp, Error, TEXT("❌ llama_decode failed."));
                continue;
            }

            n_past += embd.size();

            float* logits = llama_get_logits(ctx);
            if (!logits) continue;

            const llama_vocab* vocab = llama_model_get_vocab(model);
            int vocab_size = llama_vocab_n_tokens(vocab);
            if (vocab_size <= 0) continue;

            // Greedy sampling: choose token with max logit
            llama_token best_token = 0;
            float best_logit = logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > best_logit) {
                    best_logit = logits[i];
                    best_token = i;
                }
            }

            embd.clear();
            embd.push_back(best_token);

            std::string detok = LlamaInternal::llama_detokenize_bpe(model, embd);
            assistant_ss << detok; // accumulate assistant text
            FString AssistantText = UTF8_TO_TCHAR(assistant_ss.str().c_str());
            for (const auto& stopSeqTokens : stopSequences) {
                FString stopText;
                for (llama_token token : stopSeqTokens) {
                    // Detokenize each token individually to text
                    std::vector<llama_token> single_token{ token };
                    stopText += UTF8_TO_TCHAR(LlamaInternal::llama_detokenize_bpe(model, single_token).c_str());
                }

                if (AssistantText.EndsWith(stopText, ESearchCase::IgnoreCase)) {
                    UE_LOG(LogTemp, Warning, TEXT("🛑 Assistant text matched stop sequence: %s"), *stopText);
                    eos = true;

                    // Save the assistant reply (trimmed stop sequence if needed)
                    std::string reply = assistant_ss.str();
                    size_t pos = reply.rfind(TCHAR_TO_UTF8(*stopText));
                    if (pos != std::string::npos) {
                        reply = reply.substr(0, pos);
                    }
                    assistant_ss.str("");
                    assistant_ss << reply;

                    // Save reply to history
                    if (!reply.empty()) {
                        chatHistory.emplace_back("assistant", reply);
                    }

                    UE_LOG(LogTemp, Warning, TEXT("✅ Assistant stopped by stop sequence."));
                    continue;
                }
            }
            FString tokenOut = UTF8_TO_TCHAR(detok.c_str());
            qThreadToMain.enqueue([this, tokenOut = std::move(tokenOut)] {
                if (tokenCb) {
                    tokenCb(tokenOut);
                }
                });

            bool stop_generation = false;

            // Check if EOS token generated
            if (best_token == llama_vocab_eos(vocab)) {
                UE_LOG(LogTemp, Warning, TEXT("🛑 EOS token detected."));
                stop_generation = true;
            }
            // Optional: Detect end of assistant turn by double newlines (you can adjust)
            else if (detok.find("\n\n") != std::string::npos) {
                UE_LOG(LogTemp, Warning, TEXT("🛑 Detected double newline, assuming end of assistant reply."));
                stop_generation = true;
            }

            if (stop_generation) {
                eos = true;

                // Save assistant reply into chat history
                std::string assistantReply = assistant_ss.str();
                if (!assistantReply.empty()) {
                    chatHistory.emplace_back("assistant", assistantReply);
                }
                assistant_ss.str(""); // clear assistant text

                UE_LOG(LogTemp, Warning, TEXT("✅ Assistant answer saved. Waiting for next user input."));
                continue; // go back to sleep and wait
            }

            // Continue generation: push sampled token back
            if (n_consumed >= embd_inp.size()) {
                // Prompt is fully consumed — now autoregressive phase
                embd_inp.push_back(best_token);
            }
        }

        UE_LOG(LogTemp, Warning, TEXT("👋 LLaMA thread exiting."));
        unsafeDeactivate();
    }


    void Llama::process() {
        while (qThreadToMain.processQ()) {}
    }

    void Llama::ResetHistory() {
        chatHistory.clear();
        assistant_ss.str("");
        eos = false;
        n_consumed = 0;
        embd_inp.clear();
    }

} // namespace Internal

// --- UE LlamaComponent implementation ---

ULlamaComponent::ULlamaComponent(const FObjectInitializer& ObjectInitializer)
    : UActorComponent(ObjectInitializer), llama(std::make_unique<Internal::Llama>()) {
    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.bStartWithTickEnabled = true;

    llama->tokenCb = [this](FString NewToken) {
        OnNewTokenGenerated.Broadcast(NewToken);
        };
}

ULlamaComponent::~ULlamaComponent() = default;

void ULlamaComponent::Activate(bool bReset) {
    Super::Activate(bReset);
    LlamaInternal::Params params;
    params.prompt = prompt;
    params.pathToModel = pathToModel;
    params.stopSequences = stopSequences;
    llama->activate(bReset, std::move(params));
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

void ULlamaComponent::ResetHistory() {
    if (llama) llama->ResetHistory();
}
