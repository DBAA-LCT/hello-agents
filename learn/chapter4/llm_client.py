import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict


# 加载 .env文件中的环境变量
load_dotenv()


class HelloAgentsLLM:
    """
    定制LLM客户端：用于调用兼容OpenAI格式的接口，并默认流式相应
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提空，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        # 验证调用API的必要参数是否存在
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和基础URL不能为空")

        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)


    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大模型进行流式推理，实时打印响应内容并返回完整文本。

        Args:
            messages: 消息历史列表，每条消息为 {"role": "user"|"assistant"|"system", "content": str} 格式。
            temperature: 采样温度，控制输出随机性。取值 0~2，值越高输出越随机；默认为 0（确定性输出）。

        Returns:
            大模型返回的完整文本内容（所有流式片段拼接后的字符串）。
            若调用过程中发生异常，则返回 None。
        """
        print(f"🧠正在调用 {self.model} 模型...")
        try:
            # 发起流式 API 请求：stream=True 让响应以逐块（chunk）方式返回，
            # 从而可以边接收边打印，实现打字机效果。
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = temperature,
                stream = True,
            )

            # 处理流式响应：遍历每个 chunk，实时输出并收集完整内容
            print(" ✅大模型响应成功：")
            collected_content = []  # 用于累积所有流式片段的文本
            for chunk in response:
                # 某些 chunk（如末尾的 finish_reason）可能没有 choices，跳过它们
                if not chunk.choices:
                    continue
                # delta.content 是当前 chunk 携带的增量文本；
                # 若 chunk 没有文本字段，用空字符串兜底，避免 None 参与拼接。
                content = chunk.choices[0].delta.content or ""
                # end="" 使输出不换行，flush=True 强制立即刷新到终端，实现"打字机"效果
                print(content, end="", flush=True)
                collected_content.append(content)

            # 流式输出结束后换行，避免后续打印内容与模型输出挤在同一行
            print()
            # 将所有片段拼接成完整字符串并返回
            return "".join(collected_content)
        except Exception as e:
            print(f"❌调用LLM API时失败：{e}")
            return None

if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("------ 调用LLM-------")
        responsesText = llmClient.think(exampleMessages)
        if responsesText:
            print("\n\n------ 完整模型响应 -------")
            print(responsesText)
    except Exception as e:
        print(e)