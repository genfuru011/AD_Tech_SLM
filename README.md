# AD_Tech_SLM


##　　広告表現に特化したSLM（小規模言語モデル）の開発

## 開発環境
--HuggingFace TRL--
--Google colab [GPU=NvidiaA100]--
--Ollama--

###　手法
DPO （Direct Preference Optimization） が最適
DPO は「選ばれた文」と「却下された文」を対で学習し、相対的な優劣で判別し出力を好ましい方向に一気に寄せる

### Colab 
```python
!pip install -U transformers datasets trl peft unsloth bitsandbytes
from trl import DPOTrainer
trainer = DPOTrainer(
    model_name="LLM-NAME",
    train_dataset="ads_pref.jsonl",
    peft_config={"r":8, "lora_alpha":32, "target_modules":["q_proj","v_proj"]},
)
trainer.train()
trainer.model.save_pretrained("ad-copy-dpo")
```
## LLM選定
--Gemma3N 4B--
同規模LLMに比べ比較的性能が高く学習リソースが広いGoogle製LLM 小規模モデルながら　o4-mini並みの性能

##　必要データアセットJSONL
```json
{"prompt": "【テーマ】雨の日でもワクワクするニュースアプリを紹介してください", 
 "chosen": "雨が降っても最新トレンドをスマホでサクッとチェック！天気と話題を同時にキャッチして、移動中も退屈知らず♪", 
 "rejected": "ニュースが見られるよ！便利！"}

{"prompt": "【テーマ】忙しい会社員が移動中に英語を学べるアプリを紹介してください", 
 "chosen": "通勤電車で 3 分！AI レッスンがあなたの発音を即フィードバック💡スキマ時間で着実にスキルアップ！", 
 "rejected": "英語学習ができます。おすすめです。"}

{"prompt": "【テーマ】節電をサポートする家計簿アプリを紹介してください", 
 "chosen": "家電ごとの電気代を自動計算！グラフでムダを一目で発見して、月末の請求額にもうドキドキしません✨", 
 "rejected": "節約に役立つアプリです。ぜひ使ってください。"}
```
	•	prompt：コピーを書かせたいお題や前提条件をまとめます。
	•	chosen：CTR が高かった “良い広告テキスト” を入れます。
	•	rejected：CTR が低かった、または不採用になった文を入れます。
