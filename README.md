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

增加了Google GenAI節點
可以選擇模型，輸入API_KEY，系統提示詞、用戶提示詞就可以使用

支持的模型
"gemini-2.0-flash",

"gemini-2.0-flash-lite-preview-02-05",

"gemini-2.0-pro-exp-02-05",

"gemini-2.0-flash-thinking-exp-01-21",

"gemini-2.0-flash-exp",

"gemini-1.5-flash"

可以輸入圖片，進行反推圖片描述。可開啟是否使用Google搜尋功能

custom node缺少Google GenAI包的話 可以用這個指令安裝包

{你的ComfyUI安裝路徑}\ComfyUI_windows_portable\python_embeded\python.exe -m pip install google-genai
