#pragma once

#include <Components/ActorComponent.h>
#include <CoreMinimal.h>
#include <memory>
#include <atomic>
#include <deque>
#include <thread>
#include <functional>
#include <mutex>
#include <string>
#include <sstream>
#include <vector>
#include "llama.h"

#include "LlamaComponent.generated.h"

namespace LlamaInternal { // ✅ Proper named namespace (no anonymous)

    class Q {
    public:
        void enqueue(std::function<void()> func);
        bool processQ();

    private:
        std::deque<std::function<void()>> q;
        std::mutex mutex_;
    };

    constexpr int n_threads = 4;

    struct Params {
        FString prompt = TEXT("You are a friendly tavern keeper who welcomes adventurers.");
        FString pathToModel = TEXT("/path/to/your/model.gguf");
        TArray<FString> stopSequences;
    };

    std::vector<llama_token> my_llama_tokenize(llama_model* model, const std::string& text, std::vector<llama_token>& out, bool add_bos);
    std::string llama_detokenize_bpe(llama_model* model, const std::vector<llama_token>& tokens);

}

namespace Internal {

    class Llama {
    public:
        Llama();
        ~Llama();

        void activate(bool bReset, LlamaInternal::Params params);
        void deactivate();
        void insertPrompt(FString prompt);
        void process();
        void ResetHistory(); // ✨ Added ResetHistory

        std::function<void(FString)> tokenCb;

    private:
        llama_model* model = nullptr;
        llama_context* ctx = nullptr;

        std::vector<std::pair<std::string, std::string>> chatHistory; // ✨ Chat log
        FString systemPrompt = TEXT("You are a friendly tavern keeper who welcomes adventurers.");  // ✨ System prompt
        std::ostringstream assistant_ss;                              // ✨ Accumulate assistant tokens

        LlamaInternal::Q qMainToThread;
        LlamaInternal::Q qThreadToMain;
        std::atomic_bool running = false;
        std::thread thread;

        std::vector<llama_token> embd_inp;
        std::vector<llama_token> embd;
        std::vector<std::vector<llama_token>> stopSequences;
        std::vector<llama_token> last_n_tokens;
        int n_consumed = 0;
        int n_past = 0;
        bool eos = false;

        void threadRun();
        void unsafeActivate(bool bReset, LlamaInternal::Params params);
        void unsafeDeactivate();
        void unsafeInsertPrompt(FString prompt);
    };
}

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNewTokenGenerated, FString, NewToken);

UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class UELLAMA_API ULlamaComponent : public UActorComponent {
    GENERATED_BODY()

public:
    ULlamaComponent(const FObjectInitializer& ObjectInitializer);
    virtual ~ULlamaComponent();

    virtual void Activate(bool bReset) override;
    virtual void Deactivate() override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    UPROPERTY(BlueprintAssignable)
    FOnNewTokenGenerated OnNewTokenGenerated;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString prompt = TEXT("You are a friendly tavern keeper who welcomes adventurers.");

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString pathToModel = TEXT("/path/to/your/model.gguf");

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TArray<FString> stopSequences;

    UFUNCTION(BlueprintCallable)
    void InsertPrompt(const FString& v);

    UFUNCTION(BlueprintCallable)
    void ResetHistory(); // ✨ Added ResetHistory

private:
    std::unique_ptr<Internal::Llama> llama;
};
