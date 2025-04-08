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
#include <vector>
#include "llama.h"

#include "LlamaComponent.generated.h"

using namespace std;

namespace {
    class Q {
    public:
        void enqueue(function<void()>);
        bool processQ();

    private:
        deque<function<void()>> q;
        mutex mutex_;
    };

    void Q::enqueue(function<void()> v) {
        lock_guard l(mutex_);
        q.emplace_back(move(v));
    }

    bool Q::processQ() {
        function<void()> v;
        {
            lock_guard l(mutex_);
            if (q.empty()) return false;
            v = move(q.front());
            q.pop_front();
        }
        v();
        return true;
    }

    constexpr int n_threads = 4;

    struct Params {
        FString prompt = "Hello";
        FString pathToModel = "/path/to/your/model.gguf";
        TArray<FString> stopSequences;
    };

    vector<llama_token> my_llama_tokenize(llama_model* model, const string& text, vector<llama_token>& out, bool add_bos);
    string llama_detokenize_bpe(llama_model* model, const vector<llama_token>& tokens);
}

namespace Internal {
    class Llama {
    public:
        Llama();
        ~Llama();

        void activate(bool bReset, Params params);
        void deactivate();
        void insertPrompt(FString prompt);
        void process();

        function<void(FString)> tokenCb;

    private:
        llama_model* model = nullptr;
        llama_context* ctx = nullptr;

        Q qMainToThread;
        Q qThreadToMain;
        atomic_bool running = false;
        thread thread;

        vector<llama_token> embd_inp;
        vector<llama_token> embd;
        vector<vector<llama_token>> stopSequences;
        vector<llama_token> last_n_tokens;
        int n_consumed = 0;
        int n_past = 0;
        bool eos = false;

        void threadRun();
        void unsafeActivate(bool bReset, Params params);
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
    FString prompt = "Hello";

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString pathToModel = "/path/to/your/model.gguf";

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TArray<FString> stopSequences;

    UFUNCTION(BlueprintCallable)
    void InsertPrompt(const FString& v);

private:
    unique_ptr<Internal::Llama> llama;
};
