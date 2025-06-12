#!/usr/bin/env python3
"""
TinySwallow DPO Training - Usage Example
Google Colab環境での使用方法を示すサンプルスクリプト
"""

import json
from typing import Dict, List

def create_sample_dataset() -> List[Dict[str, str]]:
    """広告技術分野のサンプルDPOデータセットを作成"""
    return [
        {
            "prompt": "プログラマティック広告のRTBについて説明してください。",
            "chosen": "RTB（Real-Time Bidding）は、広告枠の売買をリアルタイムのオークション形式で行う仕組みです。ユーザーがWebページにアクセスした瞬間に、そのユーザーの属性や閲覧履歴に基づいて、広告主が自動的に入札を行います。最も高い金額を提示した広告主の広告が表示される仕組みで、効率的なターゲティングと費用対効果の向上を実現します。",
            "rejected": "RTBは広告を表示するシステムです。"
        },
        {
            "prompt": "DSPとSSPの違いを教えてください。",
            "chosen": "DSP（Demand-Side Platform）は広告主側のプラットフォームで、広告枠の購入を自動化し、ターゲティングや入札戦略の最適化を行います。一方、SSP（Supply-Side Platform）はメディア側のプラットフォームで、広告枠の販売を自動化し、収益の最大化を図ります。DSPは買い手、SSPは売り手の立場でプログラマティック広告の取引を支援します。",
            "rejected": "DSPとSSPは両方とも広告関連のシステムです。"
        },
        {
            "prompt": "クッキーレス時代の広告ターゲティング手法について説明してください。",
            "chosen": "クッキーレス時代では、以下の手法が重要となります：1) ファーストパーティデータの活用（自社の顧客データベース）、2) コンテキスト広告（Webページの内容に基づいたターゲティング）、3) コホート分析（類似ユーザーグループでの分析）、4) プライバシーサンドボックス技術（Googleが提案する新技術群）、5) IDソリューション（メールアドレスベースの識別子）などがあります。",
            "rejected": "クッキーが使えなくなるので、新しい方法を考える必要があります。"
        },
        {
            "prompt": "アトリビューション分析の重要性について教えてください。",
            "chosen": "アトリビューション分析は、コンバージョンに至るまでの各タッチポイントの貢献度を測定する分析手法です。ユーザーの購買行動は複雑で、複数の広告接触を経てコンバージョンに至るため、ラストクリック以外の接触点の価値も適切に評価する必要があります。これにより、マーケティング予算の最適配分、チャネル間の相互作用の理解、ROIの正確な測定が可能になります。",
            "rejected": "アトリビューション分析は広告の効果を測定する方法です。"
        },
        {
            "prompt": "ヘッダービディングとは何ですか？",
            "chosen": "ヘッダービディング（Header Bidding）は、複数のSSPやアドエクスチェンジが同時に広告枠に入札できる仕組みです。従来のウォーターフォール方式とは異なり、すべての需要源が同じタイミングで競合できるため、より公平で透明性の高いオークションが実現されます。結果として、パブリッシャーの収益向上と広告主の効率的な配信が可能になります。",
            "rejected": "ヘッダービディングは広告枠を売る方法の一つです。"
        }
    ]

def get_optimal_config_for_colab() -> Dict:
    """Google Colab環境に最適化されたトレーニング設定を返す"""
    return {
        # モデル設定
        "model_name": "SakanaAI/TinySwallow-1.5B-Instruct",
        "backup_models": [
            "tokyotech-llm/Swallow-1.5b-instruct-hf",
            "rinna/japanese-gpt-neox-3.6b-instruction-sft"
        ],
        
        # Colab最適化設定
        "per_device_train_batch_size": 1,  # メモリ制限対応
        "gradient_accumulation_steps": 8,   # 実質バッチサイズ = 8
        "learning_rate": 5e-7,              # 安定した学習
        "max_steps": 500,                   # Colab時間制限対応
        "eval_steps": 50,
        "save_steps": 100,
        "warmup_steps": 50,
        
        # DPO設定
        "beta": 0.1,
        "max_length": 1024,
        "max_prompt_length": 512,
        
        # LoRA設定（メモリ効率化）
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        
        # 量子化設定
        "use_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "use_nested_quant": False,
    }

def save_dataset_to_jsonl(dataset: List[Dict], filename: str = "sample_dpo_dataset.jsonl"):
    """データセットをJSONL形式で保存"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ データセット保存: {filename} ({len(dataset)} サンプル)")

def validate_dpo_dataset(dataset: List[Dict]) -> bool:
    """DPOデータセットの形式を検証"""
    required_keys = ['prompt', 'chosen', 'rejected']
    
    for i, item in enumerate(dataset):
        if not all(key in item for key in required_keys):
            print(f"❌ サンプル {i}: 必要なキー {required_keys} が不足")
            return False
        
        if not all(isinstance(item[key], str) for key in required_keys):
            print(f"❌ サンプル {i}: すべての値は文字列である必要があります")
            return False
    
    print(f"✅ データセット検証完了: {len(dataset)} サンプル")
    return True

def print_training_tips():
    """トレーニングのヒントを表示"""
    print("🎯 TinySwallow DPO Training - 使用のヒント")
    print("=" * 50)
    print("1. 📊 データセット準備:")
    print("   • JSONL形式でprompt, chosen, rejectedキーを含む")
    print("   • 質の高いchosen回答と明確に劣るrejected回答のペア")
    print("   • 最低100サンプル、推奨1000+サンプル")
    print()
    print("2. ⚙️ 設定の調整:")
    print("   • メモリ不足時: batch_sizeを1に、gradient_accumulation_stepsを増加")
    print("   • 学習が不安定: learning_rateを下げる (1e-7 ~ 5e-7)")
    print("   • 過学習の兆候: max_stepsを減らすかeval_stepsを頻繁に")
    print()
    print("3. 🚀 実行環境:")
    print("   • GPU必須 (T4以上推奨)")
    print("   • 実行時間: 500ステップで約30-60分")
    print("   • Colab Pro推奨 (より高性能なGPUと長時間実行)")
    print()
    print("4. 📈 評価のポイント:")
    print("   • 訓練ロスの安定した減少")
    print("   • 検証ロスの過学習チェック")
    print("   • 生成テキストの品質向上")

def main():
    """メイン実行関数"""
    print("🦆 TinySwallow DPO Training - Usage Example")
    print("=" * 50)
    
    # サンプルデータセットの作成
    dataset = create_sample_dataset()
    
    # データセット検証
    if validate_dpo_dataset(dataset):
        # JSONLファイルとして保存
        save_dataset_to_jsonl(dataset)
        
        # 最適化設定の表示
        config = get_optimal_config_for_colab()
        print(f"\n⚙️ 推奨設定:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        # 使用のヒント表示
        print()
        print_training_tips()
        
        print("\n🔗 次のステップ:")
        print("   1. Google Colabで improved notebook を開く")
        print("   2. 作成されたsample_dpo_dataset.jsonlをアップロード")
        print("   3. 設定を確認してトレーニング開始")
        print("   4. 結果を評価して必要に応じて調整")

if __name__ == "__main__":
    main()