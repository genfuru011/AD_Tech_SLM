#!/usr/bin/env python3
"""
TinySwallow DPO訓練済みモデルの評価スクリプト
広告技術ドメインでの性能評価
"""

import os
import json
import logging
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from typing import List, Dict, Any

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TinySwallowEvaluator:
    """TinySwallow DPO訓練済みモデルの評価クラス"""
    
    def __init__(self, base_model_name: str, trained_model_path: str, device: str = "auto"):
        """
        Args:
            base_model_name: ベースモデル名
            trained_model_path: DPO訓練済みモデルのパス
            device: 使用デバイス
        """
        self.base_model_name = base_model_name
        self.trained_model_path = trained_model_path
        self.device = device
        
        # M1/M2 Mac対応
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("🔧 MPS (Metal Performance Shaders) を使用")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("🔧 CUDA を使用")
            else:
                self.device = "cpu"
                logger.info("🔧 CPU を使用")
        
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """モデルとトークナイザーを読み込み"""
        logger.info(f"📥 ベースモデルを読み込み中: {self.base_model_name}")
        
        # トークナイザー読み込み
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        # パディングトークン設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # ベースモデル読み込み
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device != "mps" else torch.float32,
        }
        
        if self.device == "mps":
            # MPS環境では特別な設定
            model_kwargs["torch_dtype"] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        
        # DPO訓練済みアダプターを読み込み
        if os.path.exists(self.trained_model_path):
            logger.info(f"📥 DPO訓練済みアダプターを読み込み中: {self.trained_model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.trained_model_path)
            self.model = self.model.merge_and_unload()  # アダプターをマージ
        else:
            logger.warning(f"⚠️ 訓練済みモデルが見つかりません: {self.trained_model_path}")
            logger.info("ベースモデルを使用して評価を続行します")
        
        # デバイスに移動
        self.model.to(self.device)
        self.model.eval()
        
        # パラメータ数を表示
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"📊 総パラメータ数: {total_params:,}")
        
        logger.info("✅ モデル読み込み完了")
    
    def generate_response(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """プロンプトに対する応答を生成"""
        if not self.model or not self.tokenizer:
            raise ValueError("モデルとトークナイザーが読み込まれていません")
        
        # トークン化
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # デバイスに移動
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成設定
        generation_kwargs = {
            "max_length": max_length,
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # 応答生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs
            )
        
        # デコード
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def evaluate_on_test_prompts(self, test_prompts: List[str]) -> List[Dict[str, Any]]:
        """テストプロンプトでの評価"""
        results = []
        
        logger.info(f"📊 {len(test_prompts)}個のテストプロンプトで評価開始")
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"🔄 評価中 ({i+1}/{len(test_prompts)}): {prompt[:50]}...")
            
            try:
                response = self.generate_response(prompt)
                results.append({
                    "prompt": prompt,
                    "response": response,
                    "success": True,
                    "error": None
                })
            except Exception as e:
                logger.error(f"❌ エラー発生: {str(e)}")
                results.append({
                    "prompt": prompt,
                    "response": "",
                    "success": False,
                    "error": str(e)
                })
        
        logger.info("✅ 評価完了")
        return results
    
    def save_evaluation_results(self, results: List[Dict[str, Any]], output_path: str):
        """評価結果を保存"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # JSON形式で保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 評価結果を保存: {output_path}")
        
        # 統計情報も保存
        stats_path = output_path.replace('.json', '_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            total = len(results)
            success = sum(1 for r in results if r['success'])
            failure = total - success
            
            f.write(f"=== TinySwallow DPO 評価統計 ===\n")
            f.write(f"評価日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"総プロンプト数: {total}\n")
            f.write(f"成功: {success} ({success/total*100:.1f}%)\n")
            f.write(f"失敗: {failure} ({failure/total*100:.1f}%)\n")
            f.write(f"\n=== サンプル応答 ===\n")
            
            for i, result in enumerate(results[:3]):  # 最初の3つのサンプルを表示
                if result['success']:
                    f.write(f"\n--- サンプル {i+1} ---\n")
                    f.write(f"プロンプト: {result['prompt']}\n")
                    f.write(f"応答: {result['response']}\n")
        
        logger.info(f"📊 統計情報を保存: {stats_path}")


def main():
    """メイン評価関数"""
    # 設定
    base_model_name = "SakanaAI/TinySwallow-1.5B-Instruct"
    trained_model_path = "./outputs/tiny_swallow_dpo"  # 最新のチェックポイントを指定
    
    # チェックポイントディレクトリを検索
    checkpoint_dirs = []
    if os.path.exists(trained_model_path):
        for item in os.listdir(trained_model_path):
            if item.startswith("checkpoint-"):
                checkpoint_dirs.append(item)
    
    if checkpoint_dirs:
        # 最新のチェックポイントを使用
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
        trained_model_path = os.path.join(trained_model_path, latest_checkpoint)
        logger.info(f"🎯 最新チェックポイントを使用: {latest_checkpoint}")
    
    # 広告技術ドメインのテストプロンプト
    test_prompts = [
        "インプレッション単価(CPM)を最適化するための戦略を教えてください。",
        "リアルタイム入札(RTB)のメリットとデメリットを説明してください。",
        "コンバージョン率を向上させるためのクリエイティブ最適化手法は何ですか？",
        "プログラマティック広告におけるブランドセーフティの重要性について述べてください。",
        "アトリビューション分析を活用した広告効果測定の方法を教えてください。",
        "ヘッダービディングとウォーターフォール方式の違いを説明してください。",
        "DSPとSSPの役割と連携について教えてください。",
        "ビューアビリティ測定の課題と解決策について説明してください。",
        "フリークエンシーキャップの設定方法と効果を教えてください。",
        "CookieレスWorld対応のための準備について述べてください。"
    ]
    
    # 評価器初期化
    evaluator = TinySwallowEvaluator(base_model_name, trained_model_path)
    
    try:
        # モデル読み込み
        evaluator.load_model()
        
        # 評価実行
        results = evaluator.evaluate_on_test_prompts(test_prompts)
        
        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./outputs/evaluation/tiny_swallow_evaluation_{timestamp}.json"
        evaluator.save_evaluation_results(results, output_path)
        
        logger.info("🎉 評価が正常に完了しました！")
        
    except Exception as e:
        logger.error(f"❌ 評価中にエラーが発生しました: {str(e)}")
        raise


if __name__ == "__main__":
    main()
