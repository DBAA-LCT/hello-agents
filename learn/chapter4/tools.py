import os
from serpapi import SerpApiClient
from typing import Dict, Any
import ast
import operator





# 运算符白名单：左边是 ast 用来表示运算符的类型，右边是实际执行运算的函数。
# 例如，识别到乘法 ast.Mult 时，就调用 operator.mul(left, right)。
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,

}

# ast.parse() 会把表达式拆成树状结构。下面是几种简化示意：
#
# 表达式 5：                    表达式 -5：
# Constant(value=5)             UnaryOp
#                                ├─ op: USub（负号）
#                                └─ operand: Constant(value=5)
#
# 表达式 2 + 3 * 4：
# BinOp（最外层是加法）
# ├─ left: Constant(value=2)
# ├─ op: Add（加号）
# └─ right: BinOp（右边是乘法，因为乘法优先）
#    ├─ left: Constant(value=3)
#    ├─ op: Mult（乘号）
#    └─ right: Constant(value=4)
#
# 因此 node.left / node.op / node.right 是 BinOp 节点的三个部分；
# node.operand / node.op 是 UnaryOp 节点的两个部分。

def _evaluate_math(node):
    """计算语法树中的一个节点；node 是当前正在处理的表达式部分。"""

    # ast.Constant 表示一个固定值，例如表达式 "5" 中的数字 5。
    if isinstance(node, ast.Constant):
        # bool 是 int 的子类，但 True/False 不是我们希望接受的数学输入。
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("表达式里只能使用数字。")
        # node.value 是这个常量节点里实际保存的数字。
        return node.value

    # ast.UnaryOp 表示一元运算：运算符只作用于一个值，例如 -5 或 +5。
    if isinstance(node, ast.UnaryOp):
        # node.operand 是正负号后面的值；它本身也可能是一个更复杂的表达式。
        value = _evaluate_math(node.operand)
        # node.op 表示具体的一元运算符：UAdd 是 +，USub 是 -。
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("不支持这个一元运算符。")

    # ast.BinOp 表示二元运算：运算符连接左、右两个值，例如 2 + 3。
    if isinstance(node, ast.BinOp):
        # node.op 是运算符类型（如 ast.Add、ast.Mult）；用它从白名单中查出计算函数。
        operation = _ALLOWED_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("只支持加减乘除运算。")

        # node.left 和 node.right 是运算符两侧的节点。
        # 递归处理它们，是为了支持 2 + 3 * 4 这类嵌套表达式和括号。
        left = _evaluate_math(node.left)
        right = _evaluate_math(node.right)
        # 例如 operation 是 operator.add 时，这行就相当于 left + right。
        return operation(left, right)

    # 其他节点类型（例如函数调用、变量名）一律拒绝，避免执行任意 Python 代码。
    raise ValueError("表达式中包含不支持的内容。")

def calculator(expression: str) -> str:
    """计算只包含数字和四则运算的数学表达式，并返回字符串结果。"""
    try:
        # 将输入字符串解析成语法树；mode="eval" 表示只接受一个表达式。
        tree = ast.parse(expression, mode="eval")
        # tree.body 是表达式的根节点，从这里开始逐层检查并计算。
        result = _evaluate_math(tree.body)
        # 工具约定返回文本，所以把数字结果转成字符串。
        return str(result)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return f"计算失败：{error}"
        
def search(query: str) -> str:
    """
    一个jiyu SerpApi的实战网页搜索引擎工具。
    它会只能地分析结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 设置搜索区域为中国
            "hl": "zh",  # 设置搜索语言为中文
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 优先返回直接答案，再返回知识图谱描述，最后返回常规搜索摘要。
        answer_box_list = results.get("answer_box_list") or []
        if answer_box_list:
            return "\n".join(str(answer) for answer in answer_box_list)

        answer_box = results.get("answer_box") or {}
        if isinstance(answer_box, dict) and answer_box.get("answer"):
            return str(answer_box["answer"])

        knowledge_graph = results.get("knowledge_graph") or {}
        if isinstance(knowledge_graph, dict) and knowledge_graph.get("description"):
            return str(knowledge_graph["description"])

        organic_results = results.get("organic_results") or []
        if organic_results:
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(organic_results[:3])
                if isinstance(res, dict)
            ]
            if snippets:
                return "\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索是发生错误：{e}"



class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}


    def registerTool(self, name: str, description: str, func: callable):
        """
        像工具箱中注册一个新工具
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据函数名获取一个工具的执行函数
        """
        return self.tools.get(name, {}).get("func")

    def getAvaliableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
    
