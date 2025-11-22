#!/usr/bin/env python3
import os
import sys
import httpx  # 用于配置 HTTP 客户端
from openai import OpenAI
from pathlib import Path
# 这里程序根据openAI 2.8.1版本写，不同版本的openAI模块API可能差距很大
# 注意openai模块需要用pip安装，安装之前请在当前目录下设置虚拟环境
# 设置虚拟环境：python -m venv .venv
# 激活虚拟环境：source .venv/bin/activate
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# 从系统环境变量里面获取OPENAI_API_KEY，可以用export OPENAI_API_KEY=直接设置一个临时的
# 需要在系统环境变量里面设置OPENAI_API_KEY为openAI的对应API key
if not OPENAI_API_KEY: # OPENAI_API_KEY没有设置就退出
    print("请先在环境变量中设置 OPENAI_API_KEY（或使用 .env 文件并加载）。")
    sys.exit(1)# 异常退出，状态码为1

# 创建 OpenAI 客户端对象
# 如果你的代理是 HTTP/HTTPS 协议，可以直接使用
# 如果是 SOCKS 代理，需要转换成 HTTP 代理或使用 httpx-socks
client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(
        proxy="http://127.0.0.1:7890",  # 这里是VPN代理，
        timeout=30.0  # 设置 30 秒超时，避免网络慢时过早失败
    )
)
def get_user_input():
    # 优先从命令行参数获取
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    # 运行程序的时候一般是输入python main.py ... ，这里sys.argv[0]是main.py
    # join的作用是用" "作为分隔符把后面的命令连成一串，也就是说这里支持之间从命令行传入用户的输入
    # 否则交互式读取 stdin
    print("请输入要发送给 AI 的内容（结束后按 Enter）：")
    return sys.stdin.readline().strip()
    # .strip() - 去除首尾空白字符，基本上处理用户的输入都需要这个

def call_chat_model(prompt: str, model: str = "gpt-4o-mini"):  # 这个最便宜
    # 新版 API 调用方式
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        # 1token大约为0.75英文单词或者0.5~0.7个汉字，也是计费单位
        temperature=0.7,
        # 随机性/创造性参数，范围0～2,越大随机性越强
    )
    return response.choices[0].message.content.strip()
"""
OpenAI API (>= 1.0.0)的主要使用方式：
导入方式：from openai import OpenAI
创建客户端：client = OpenAI(api_key=...)
调用方式：client.chat.completions.create(...)
返回值：使用属性访问 response.choices[0].message.content

create方法会自动把参数打包成.json格式，构建一个HTTP请求，发送到OpenAI的服务器，
返回值存放在response中

response本质是一个对象，结构如下（属性值是随便取的）：
    response.id = "chatcmpl-abc123"
    response.object = "chat.completion"
    response.created = 1234567890
    response.model = "gpt-3.5-turbo"
    response.choices = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "你好！我是 AI 助手，很高兴认识你。"
            },
            "finish_reason": "stop"
        }
    ]
    response.usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
"""

def main():
    prompt = get_user_input()
    if not prompt:
        print("没有输入，退出。")
        return

    try:
        ai_output = call_chat_model(prompt)
    except Exception as e:
        print(f"调用 API 出错：{type(e).__name__}: {e}")
        # 打印详细的错误信息，便于调试
        import traceback
        print("\n详细错误堆栈：")
        traceback.print_exc()
        return
    # 处理异常，尤其是对外接口，以后实现项目一定要注意，每一个接口只能有规定的输入输出，若不是要有异常处理
    print("AI 返回：")
    print(ai_output)

    out_path = Path("ai_output.txt")
    # 这里相当于C语言的FILE* out_path=fopen(...)，表示打开一个文件对象
    # 输出内容放在当前目录下的ai_output.txt文件
    out_path.write_text(ai_output, encoding="utf-8")
    print(f"已将 AI 输出写入 {out_path.resolve()}")

if __name__ == "__main__":
    main()