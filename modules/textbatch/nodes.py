import os
import json
import logging
from server import PromptServer
import torch
import numpy as np
from typing import List, Union
from qwen_vl_utils import process_vision_info
from PIL import Image
import folder_paths

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

class QwenLocalNode:
    """
    Qwen 本地模型節點
    使用本地的 Qwen 模型進行文本生成
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bf16_support = torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Enabled",
                    "label_off": "Disabled"
                }),
                "model_size": ([
                    "0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"
                ], {
                    "default": "3B"
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
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 16384,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("response_text", "filtered_text", "think_content", "status")
    FUNCTION = "generate"
    CATEGORY = "TextBatch"

    def get_model_id(self, model_size):
        """根據模型大小獲取對應的模型 ID"""
        size_to_id = {
            "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
            "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
            "3B": "Qwen/Qwen2.5-3B-Instruct",
            "7B": "Qwen/Qwen2.5-7B-Instruct",
            "14B": "Qwen/Qwen2.5-14B-Instruct",
            "32B": "Qwen/Qwen2.5-32B-Instruct",
            "72B": "Qwen/Qwen2.5-72B-Instruct"
        }
        return size_to_id.get(model_size)

    def get_model_path(self, model_size):
        """根據模型大小獲取本地模型路徑"""
        import os
        from huggingface_hub import snapshot_download

        # 獲取模型 ID
        model_id = self.get_model_id(model_size)
        # 構建本地路徑，使用 ComfyUI 的標準模型目錄
        model_dir = os.path.join(folder_paths.models_dir, "LLM", os.path.basename(model_id))
        
        # 如果模型不存在，下載模型
        if not os.path.exists(model_dir):
            logger.info(f"正在從 Hugging Face 下載模型到 {model_dir}")
            snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
        
        return model_dir

    def load_model(self, model_size):
        """加載模型和分詞器"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not self.model_loaded:
                # 獲取模型路徑
                model_dir = self.get_model_path(model_size)

                # 設定 GPU 記憶體使用
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                max_memory = {0: f"{int(gpu_memory * 0.9 / 1024**3)}GB"}  # 使用 90% 的 GPU 記憶體

                logger.info(f"正在加載分詞器，路徑: {model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

                logger.info(f"正在加載模型，路徑: {model_dir}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map="auto",
                    max_memory=max_memory,
                )
                self.model_loaded = True
                return True, "模型加載成功"
            return True, "模型已加載"
        except Exception as e:
            logger.error(f"模型加載錯誤: {str(e)}")
            return False, f"模型加載失敗: {str(e)}"

    def filter_think_content(self, text):
        """過濾掉文本中的 think 標籤及其內容"""
        import re
        filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        filtered_text = re.sub(r'\n\s*\n', '\n', filtered_text)
        return filtered_text.strip()

    def extract_think_content(self, text):
        """提取文本中的 think 標籤內容"""
        import re
        think_contents = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        return '\n'.join(think_contents).strip()

    def generate(self, enable, model_size, system_prompt, prompt, max_new_tokens, temperature):
        try:
            if not enable:
                return ("", "", "", "節點已禁用")

            # 加載模型
            success, message = self.load_model(model_size)
            if not success:
                return ("", "", "", message)

            # 準備消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            # 使用 tokenizer 的 chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 準備模型輸入
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # 生成回應
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # 只保留新生成的部分
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # 解碼回應
            response = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            # 處理輸出
            filtered_text = self.filter_think_content(response)
            think_content = self.extract_think_content(response)
            
            return (response, filtered_text, think_content, "生成成功")

        except Exception as e:
            logger.error(f"Qwen 本地模型錯誤: {str(e)}")
            return ("", "", "", f"錯誤: {str(e)}")

class QwenVLNode:
    """
    Qwen VL 多模態模型節點
    使用本地的 Qwen VL 模型進行圖文生成
    """
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.bf16_support = torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Enabled",
                    "label_off": "Disabled"
                }),
                "model_size": ([
                    "3B", "7B"
                ], {
                    "default": "3B"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.Describe this picture",
                    "placeholder": "輸入系統提示詞"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "輸入用戶提示詞"
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response_text", "status")
    FUNCTION = "generate"
    CATEGORY = "TextBatch"

    def tensor_to_base64(self, image_tensor, batch_index=0):
        """將 tensor 轉換為 base64 字符串"""
        import base64
        import io

        # ComfyUI中的圖像格式是 BCHW (Batch, Channel, Height, Width)
        if len(image_tensor.shape) == 4:  # BCHW format
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)  # 移除batch維度
        
        # 確保值範圍在[0,1]之間並轉換為uint8
        image = (torch.clamp(image_tensor, 0, 1) * 255).cpu().numpy().astype(np.uint8)
        
        # 轉換為PIL圖像
        pil_image = Image.fromarray(image, mode='RGB')
        
        # 將PIL圖像轉換為base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"

    def get_model_id(self, model_size):
        """根據模型大小獲取對應的模型 ID"""
        size_to_id = {
            "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
            "7B": "Qwen/Qwen2.5-VL-7B-Instruct"
        }
        return size_to_id.get(model_size)

    def get_model_path(self, model_size):
        """根據模型大小獲取本地模型路徑"""
        import os
        from huggingface_hub import snapshot_download

        # 獲取模型 ID
        model_id = self.get_model_id(model_size)
        # 構建本地路徑，使用 ComfyUI 的標準模型目錄
        model_dir = os.path.join(folder_paths.models_dir, "LLM", os.path.basename(model_id))
        
        # 如果模型不存在，下載模型
        if not os.path.exists(model_dir):
            logger.info(f"正在從 Hugging Face 下載模型到 {model_dir}")
            snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
        
        return model_dir

    def load_model(self, model_size):
        """加載模型和處理器"""
        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoTokenizer,
                AutoProcessor,
                BitsAndBytesConfig
            )
            import torch

            if not self.model_loaded:
                # 獲取模型路徑
                model_dir = self.get_model_path(model_size)

                # 設定量化配置
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                logger.info(f"正在加載處理器和分詞器，路徑: {model_dir}")
                self.processor = AutoProcessor.from_pretrained(
                    model_dir,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    trust_remote_code=True
                )

                logger.info(f"正在加載模型，路徑: {model_dir}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_dir,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.model_loaded = True
                return True, "模型加載成功"
            return True, "模型已加載"
        except Exception as e:
            logger.error(f"模型加載錯誤: {str(e)}")
            return False, f"模型加載失敗: {str(e)}"

    def filter_think_content(self, text):
        """過濾掉文本中的 think 標籤及其內容"""
        import re
        filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        filtered_text = re.sub(r'\n\s*\n', '\n', filtered_text)
        return filtered_text.strip()

    def extract_think_content(self, text):
        """提取文本中的 think 標籤內容"""
        import re
        think_contents = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        return '\n'.join(think_contents).strip()

    def generate(self, enable, model_size, system_prompt, prompt, max_new_tokens, temperature, image=None):
        try:
            if not enable:
                return ("", "節點已禁用")

            # 加載模型
            success, message = self.load_model(model_size)
            if not success:
                return ("", message)

            with torch.no_grad():
                # 準備消息
                if image is not None:
                    base64_image = self.tensor_to_base64(image)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": base64_image,
                                },
                                {
                                    "type": "text", 
                                    "text": prompt
                                },
                            ],
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]

                # 準備推理輸入
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # 處理視覺信息
                image_inputs, video_inputs = process_vision_info(messages)

                # 處理輸入
                inputs = self.processor(
                    text=[text],  # 注意這裡要用列表
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                # 生成回應
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True
                )

                # 只保留新生成的部分
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]

                # 解碼回應
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                return (response, "生成成功")

        except Exception as e:
            logger.error(f"Qwen VL 模型錯誤: {str(e)}")
            return ("", f"錯誤: {str(e)}")

# 更新節點類映射
NODE_CLASS_MAPPINGS = {
    "TogetherAI": TogetherAINode,
    "ThinkR1TextSplitter": ThinkR1TextSplitter,
    "QwenLocal": QwenLocalNode,
    "QwenVL": QwenVLNode
}

# 更新節點顯示名稱映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "TogetherAI": "Together AI",
    "ThinkR1TextSplitter": "Think R1 Text Splitter",
    "QwenLocal": "Qwen Local",
    "QwenVL": "Qwen VL"
}