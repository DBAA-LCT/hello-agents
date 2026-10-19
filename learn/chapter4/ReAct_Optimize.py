import json

from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE= """
请只返回一个合法的JSON对象，不要添加Markdown代码围栏或其他说明。

需要调用工具时，返回：
{{
    "thought": "简短说明下一步要做什么",
    "action": {{
        "type": "tool",
        "name": "工具名称",
        "input": "传给工具的内容"
    }}
}}

已经能够回答问题时，返回：
{{
    "thought": "简短说明为什么可以结束",
    "action": {{
        "type": "finish",
        "answer": "给用户的最终答案"
    }}
}}


可用工具：
{tools}

用户问题：
{question}

历史记录：
{history}
"""


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行是重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"---第{current_step}步---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvaliableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            thought, action = None, None
            for parse_attempt in range(2):
                response_text = self.llm_client.think(messages=messages)
                if response_text:
                    thought, action = self._parse_output(response_text)

                if action:
                    break

                if parse_attempt == 0:
                    if response_text:
                        messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "上一次回复无法解析为要求的 JSON。请只返回一个合法 JSON 对象，"
                            "并严格使用提示词中规定的字段和结构。"
                        ),
                    })

            if not action:
                print("错误：LLM两次都未能返回有效的JSON Action，流程终止。")
                break

            if thought:
                print(f"🤔思考：{thought}")

            # 4. 根据action的类型执行操作
            action_type = action.get("type")

            if action_type == "finish":
                final_answer = action.get("answer")
                if not isinstance(final_answer, str):
                    print("错误：finish action 缺少有效的 answer。")
                    break
                print(f"🎉最终答案：{final_answer}")
                return final_answer

            if action_type != "tool":
                print(f"错误：无法识别的action类型：{action_type}")
                break


            tool_name = action.get("name")
            tool_input = action.get("input")

            if not isinstance(tool_name, str) or not isinstance(tool_input, str):
                print(f"错误：tool action缺少有效的name或input。")
                break


            print(f"🎬行动：{tool_name}[{tool_input}]")


            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误：未找到名为 '{tool_name}'的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具

            print(f"👀 观察：{observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {json.dumps(action, ensure_ascii=False)}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None


    def _parse_output(self, text: str):
        """
        把模型返回的JSON文本解析成 thought 和action。
        """
        text = text.strip()

        # 兼容模型偶尔用 Markdown 代码围栏包裹 JSON 的情况。
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
                if text.lower().startswith("json"):
                    text = text[4:].lstrip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as error:
            print(f"JSON解析失败：{error}")
            return None, None

        if not isinstance(data, dict):
            return None, None

        thought = data.get("thought")
        action = data.get("action")

        if not isinstance(thought, str) or not isinstance(action, dict):
            return None, None

        return thought, action



if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
