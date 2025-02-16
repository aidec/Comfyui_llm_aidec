# Comfyui_llm_aidec
針對together.xyz寫的api，主要是comfyui_llm_party 這個似乎不支援，就簡單實現一下。

在Comfyui搜尋 Together AI 節點
填入API_KEY，輸入要使用的模型，系統提示詞、用戶提示詞就可以使用

安裝方式在ComfyUI\custom_nodes 將此專案git clone下來

Together AI支持的模型都可以使用，像是
DeepSeek R1
DeepSeek V3
Meta Llama 3.3 70B Instruct Turbo
Meta Llama 3.1 405B Instruct Turbo

完整清單
https://api.together.xyz/models

comfyui-deepseek-r1-llm-together測試文
https://blog.aidec.tw/post/comfyui-deepseek-r1-llm-together

增加Qwen2.5-VL、Qwen2.5-Instruct 兩個節點
可以用來簡單LLM詢問
跟輸入圖片進行反推圖片描述
