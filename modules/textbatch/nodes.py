import os
import json
import logging
from server import PromptServer
import torch
import numpy as np
from typing import List, Union

# 設定基本的日誌記錄格式和級別
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TogetherAINode:
    """
    Together AI API 節點
    用於調用 Together AI 的語言模型 API，支援 chat completions
    """
    def __init__(self):
        self.client = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Enabled",
                    "label_off": "Disabled"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "輸入你的 Together API Key"
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "deepseek-ai/DeepSeek-R1",
                    "placeholder": "輸入模型名稱"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "placeholder": "輸入系統提示詞"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "輸入用戶提示詞"
                }),
                "max_tokens": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("response_text", "filtered_text", "think_content", "status")
    FUNCTION = "generate"
    CATEGORY = "TextBatch"

    def filter_think_content(self, text):
        """過濾掉文本中的 think 標籤及其內容"""
        import re
        # 使用正則表達式移除 <think> 標籤及其內容
        filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # 移除多餘的空行
        filtered_text = re.sub(r'\n\s*\n', '\n', filtered_text)
        return filtered_text.strip()

    def extract_think_content(self, text):
        """提取文本中的 think 標籤內容"""
        import re
        think_contents = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        # 將所有 think 內容合併，用換行符分隔
        return '\n'.join(think_contents).strip()

    def generate(self, enable, api_key, model, system_prompt, prompt, max_tokens, temperature):
        try:
            # 如果節點被禁用，直接返回空結果
            if not enable:
                return ("", "", "", "Node is disabled")

            # 初始化 OpenAI 客戶端
            import openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1",
            )

            # 準備消息列表
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            # 調用 chat completions API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # 提取生成的文本
            generated_text = response.choices[0].message.content
            
            # 過濾後的文本
            filtered_text = self.filter_think_content(generated_text)
            
            # 提取 think 內容
            think_content = self.extract_think_content(generated_text)
            
            status = "生成成功"

            return (generated_text, filtered_text, think_content, status)

        except Exception as e:
            logger.error(f"Together AI API 調用錯誤: {str(e)}")
            return ("", "", "", f"錯誤: {str(e)}")

class ThinkR1TextSplitter:
    """
    文本分離節點
    用於分離含有 think 標籤的文本，將其分為完整文本、過濾後文本和思考內容
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "輸入包含 <think> 標籤的文本"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("response_text", "filtered_text", "think_content")
    FUNCTION = "process"
    CATEGORY = "TextBatch"

    def filter_think_content(self, text):
        """過濾掉文本中的 think 標籤及其內容"""
        import re
        # 使用正則表達式移除 <think> 標籤及其內容
        filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # 移除多餘的空行
        filtered_text = re.sub(r'\n\s*\n', '\n', filtered_text)
        return filtered_text.strip()

    def extract_think_content(self, text):
        """提取文本中的 think 標籤內容"""
        import re
        think_contents = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        # 將所有 think 內容合併，用換行符分隔
        return '\n'.join(think_contents).strip()

    def process(self, text):
        try:
            # 原始文本
            response_text = text
            
            # 過濾後的文本
            filtered_text = self.filter_think_content(text)
            
            # 提取 think 內容
            think_content = self.extract_think_content(text)
            
            return (response_text, filtered_text, think_content)

        except Exception as e:
            logger.error(f"文本處理錯誤: {str(e)}")
            return ("", "", "")

# 更新節點類映射
NODE_CLASS_MAPPINGS = {
    "TogetherAI": TogetherAINode,
    "ThinkR1TextSplitter": ThinkR1TextSplitter
}

# 更新節點顯示名稱映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherAI": "Together AI",
    "ThinkR1TextSplitter": "Think R1 Text Splitter"
}