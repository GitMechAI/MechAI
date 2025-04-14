# mechanical_assistant.py
import os
import httpx
import streamlit as st
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel, Field  # 确保正确导入
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser


# --------------------------
# 正确定义的Pydantic模型
# --------------------------
class MechanicalDesign(BaseModel):  # 显式继承BaseModel
    components: List[str] = Field(
        description="设计组件清单",
        examples=[["液压缸", "行星齿轮组"]]
    )
    materials: Dict[str, str] = Field(
        description="材料性能参数",
        examples=[{"齿轮": "20CrMnTi", "壳体": "QT600-3"}]
    )
    parameters: Dict[str, float] = Field(
        description="关键设计参数",
        examples=[{"额定扭矩": 320.5, "传动效率": 0.92}]
    )


# --------------------------
# 模型处理链（修复版本）
# --------------------------
def create_llm_chain(api_key: str, proxy_url: str = None):
    """创建兼容Pydantic V1的处理链"""
    client_params = {
        "model": "moonshot-v1-8k",
        "openai_api_key": api_key,
        "openai_api_base": "https://api.moonshot.cn/v1",
        "temperature": 0.7
    }

    if proxy_url:
        client_params["http_client"] = httpx.Client(
            proxies=proxy_url,
            transport=httpx.HTTPTransport(retries=3)
        )

    llm = ChatOpenAI( ** client_params)

    # 构建提示模板
    system_template = """作为机械设计专家，请完成：
1. 根据{standard}标准分析需求
2. 推荐材料并验证力学性能
3. 输出三维方案框架

{format_instructions}"""

    # 创建解析器（关键修复点）
    parser = PydanticOutputParser(pydantic_object=MechanicalDesign)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "设计需求：{requirement}")
    ])

    return prompt | llm | parser


# --------------------------
# Streamlit交互界面
# --------------------------
def main():
    st.set_page_config(page_title="智能机械设计平台", layout="wide", page_icon="⚙️")

    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []

    # 侧边栏设置
    with st.sidebar:
        st.header("🔑 配置中心")
        api_key = st.text_input("Moonshot API密钥", type="password")

        # 代理设置
        use_proxy = st.checkbox("启用代理")
        proxy_url = st.text_input("代理地址", "http://127.0.0.1:7890") if use_proxy else None

        # 历史记录
        st.divider()
        st.header("📜 方案历史")
        for item in reversed(st.session_state.history[-3:]):
            st.caption(f"{item['time']}")
            st.code(item["summary"], language="json")

    # 主界面
    st.title("🔧 智能机械设计系统")

    # 输入区域
    with st.form(key="design_form"):
        requirement = st.text_area(
            "设计需求描述：",
            height=200,
            placeholder="例：需要设计承载5吨的液压升降平台，升降行程2米...",
            help="包含功能需求和技术参数"
        )

        col1, col2 = st.columns(2)
        with col1:
            standard = st.selectbox("设计标准", ["GB", "ISO", "ASME"], index=0)

        submitted = st.form_submit_button("🚀 生成方案", disabled=not api_key)

    # 处理生成请求
    if submitted:
        if not requirement.strip():
            st.error("⚠️ 请输入设计需求内容")
            st.stop()

        if not api_key:
            st.error("⚠️ 请先输入有效的API密钥")
            st.stop()

        with st.spinner("AI正在生成方案..."):
            try:
                # 创建处理链
                chain = create_llm_chain(api_key, proxy_url)

                # 获取格式指令
                parser = PydanticOutputParser(pydantic_object=MechanicalDesign)

                # 执行链式调用
                result = chain.invoke({
                    "standard": standard,
                    "requirement": requirement,
                    "format_instructions": parser.get_format_instructions()
                })

                # 保存结果
                st.session_state.current_result = result
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "summary": f"标准：{standard}\n组件：{result.components[:2]}..."
                })

            except Exception as e:
                st.error(f"生成失败：{str(e)}")
                st.stop()

    # 显示结果
    if "current_result" in st.session_state:
        result = st.session_state.current_result

        with st.expander("📋 设计方案详情", expanded=True):
            col1, col2 = st.columns([2, 3])

            with col1:
                st.subheader("🛠️ 组件清单")
                st.table({"组件名称": result.components})

                st.subheader("🔩 推荐材料")
                st.table(list(result.materials.items()))

            with col2:
                st.subheader("⚖️ 技术参数")
                st.json(result.parameters, expanded=True)


if __name__ == "__main__":
    main()
