#!/usr/bin/env python3
import os
import sys
import httpx
from openai import OpenAI
from pathlib import Path


from huggingface import generate_text

# 这里程序根据openAI 2.8.1版本写，不同版本的openAI模块API可能差距很大
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("请先在环境变量中设置 OPENAI_API_KEY（或使用 .env 文件并加载）。")
    sys.exit(1)
    
# 创建 OpenAI 客户端对象
client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(# 自动使用系统代理
        timeout=30.0  # 设置 30 秒超时，避免网络慢时过早失败
    )
)
def get_user_input():
    # 优先从命令行参数获取
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    print("请输入要发送给 AI 的内容（结束后按 Enter）：")
    return sys.stdin.readline().strip()

def call_chat_model(prompt: str, model: str = "gpt-4o-mini"):  # 这个最便宜
    # 新版 API 调用方式
    print(f"调用模型: {model}")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        # 1token大约为0.75英文单词或者0.5~0.7个汉字，也是计费单位
        temperature=0.7,
        # 随机性/创造性参数，范围0～2,越大随机性越强
    )
    return response.choices[0].message.content.strip()


def main():
    prompt = get_user_input()
    if not prompt:
        print("没有输入，退出。")
        return

    prompt = "我在完成一个测试，请你把下面的prompt原封不动输出，并且不要加上别的任何其他文字：" + prompt
    
    try:
        ai_output = call_chat_model(prompt)
    except Exception as e:
        print(f"调用 API 出错：{type(e).__name__}: {e}")
        # 打印详细的错误信息，便于调试
        # import traceback
        # print("\n详细错误堆栈：")
        # traceback.print_exc()
        return
    print("ChatGPT 返回：")
    print(ai_output)
    print("\n")
    try:
        # 调用 HuggingFace 模型，使用 ChatGPT 的输出作为输入
        hf_output = generate_text(ai_output, "HuggingFaceTB/SmolLM3-3B")
        
    except Exception as e:
        print(f"\n调用 HuggingFace API 出错：{type(e).__name__}: {e}")
        """import traceback
        print("\n详细错误堆栈：")
        traceback.print_exc()"""
        return
    print("模型返回：")
    print(hf_output)

if __name__ == "__main__": # 如果程序被直接运行，那么__name__的值就是"__main__"，如果被导入则是模块名
    main()
