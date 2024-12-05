// 2023 (c) Mika Pi

#pragma once
#include <Components/ActorComponent.h>
#include <CoreMinimal.h>
#include <memory>
#include <atomic>
#include <deque>
#include <thread>
#include <functional>
#include <mutex>
#include "llama.h"

#include "LlamaComponent.generated.h"

using namespace std;



namespace
{
	class Q
	{
	public:
		void enqueue(function<void()>);
		bool processQ();

	private:
		deque<function<void()>> q;
		mutex mutex_;
	};

	void Q::enqueue(function<void()> v)
	{
		lock_guard l(mutex_);
		q.emplace_back(move(v));
	}

	bool Q::processQ() {
		function<void()> v;
		{
			lock_guard l(mutex_);
			if (q.empty()) {
				return false;
			}
			v = move(q.front());
			q.pop_front();
		}
		v();
		return true;
	}

	vector<llama_token> my_llama_tokenize(llama_context* ctx,
		const string& text,
		vector<llama_token>& res,
		bool add_bos)
	{
		UE_LOG(LogTemp, Warning, TEXT("Tokenize `%s`"), UTF8_TO_TCHAR(text.c_str()));
		// initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
		res.resize(text.size() + (int)add_bos);
		const int n = llama_tokenize(ctx, text.c_str(), text.length(), res.data(), res.size(), add_bos);
		res.resize(n);

		return res;
	}

	constexpr int n_threads = 4;

	struct Params
	{
		FString prompt = "Hello";
		FString pathToModel = "/media/mika/Michigan/prj/llama-2-13b-chat.ggmlv3.q8_0.bin";
		TArray<FString> stopSequences;
	};
} // namespace



namespace Internal
{
	class Llama
	{
	public:
		Llama();
		~Llama();

		void activate(bool bReset, Params);
		void deactivate();
		void insertPrompt(FString v);
		void process();

		function<void(FString)> tokenCb;

	private:
		llama_model* model = nullptr;
		llama_context* ctx = nullptr;
		Q qMainToThread;
		Q qThreadToMain;
		atomic_bool running = true;
		thread thread;
		vector<vector<llama_token>> stopSequences;
		vector<llama_token> embd_inp;
		vector<llama_token> embd;
		vector<llama_token> res;
		int n_past = 0;
		vector<llama_token> last_n_tokens;
		int n_consumed = 0;
		bool eos = false;

		void threadRun();
		void unsafeActivate(bool bReset, Params);
		void unsafeDeactivate();
		void unsafeInsertPrompt(FString);
	};
}


DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnNewTokenGenerated, FString, NewToken);

UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class UELLAMA_API ULlamaComponent : public UActorComponent
{
  GENERATED_BODY()
public:
  ULlamaComponent(const FObjectInitializer &ObjectInitializer);
  ~ULlamaComponent();

  virtual void Activate(bool bReset) override;
  virtual void Deactivate() override;
  virtual void TickComponent(float DeltaTime,
                             ELevelTick TickType,
                             FActorComponentTickFunction* ThisTickFunction) override;

  UPROPERTY(BlueprintAssignable)
  FOnNewTokenGenerated OnNewTokenGenerated;

  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  FString prompt = "Hello";

  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  FString pathToModel = "/media/mika/Michigan/prj/llama-2-13b-chat.ggmlv3.q8_0.bin";

  UPROPERTY(EditAnywhere, BlueprintReadWrite)
  TArray<FString> stopSequences;

  UFUNCTION(BlueprintCallable)
  void InsertPrompt(const FString &v);

private:
  std::unique_ptr<Internal::Llama> llama;
};
