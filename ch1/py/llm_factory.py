#!/usr/bin/env python3
"""
LLM 工厂类 - 统一管理 ChatOpenAI 实例
支持多种模型配置，便于切换和管理
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

"""
deepseek-ai/DeepSeek-R1
deepseek-ai/DeepSeek-V3

baidu/ERNIE-4.5-300B-A47B
tencent/Hunyuan-A13B-Instruct


THUDM/GLM-4.1V-9B-Thinking
"""


# 加载环境变量
load_dotenv()

class LLMFactory:
    """LLM 工厂类，用于创建和管理不同的 ChatOpenAI 实例"""
    
    # 预定义的模型配置
    MODEL_CONFIGS = {
        # 阿里云 DashScope 模型
        "qwen-plus": {
            "model": "qwen-plus",
            "api_key": os.getenv("DASHSCOPE_API_KEY", "sk-4000a9e2f25d469fb524a5174df01693"),
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        },
        "qwen-turbo": {
            "model": "qwen-turbo",
            "api_key": os.getenv("DASHSCOPE_API_KEY", "sk-4000a9e2f25d469fb524a5174df01693"),
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        },
        
        # 月之暗面 Kimi 模型
        "kimi-k2": {
            "model": "kimi-k2-0711-preview",
            "api_key": os.getenv("MOONSHOT_API_KEY", "sk-mGrUVkP78Nbqt83f8o25s3pi2WgtvSuOMvK2hXZXDWxmfnOl"),
            "api_base": "https://api.moonshot.cn/v1"
        },
        # 硅基流动 模型 - 支持动态模型选择
        "siliconflow": {
            "api_key": os.getenv("SILICONFLOW_API_KEY", "sk-dsggnssrlfjsubmypbflbmtijrfcpjkjfcqmbhbpulnxukyg"),
            "api_base": "https://api.siliconflow.cn/v1",
            "dynamic_model": True  # 标记为动态模型
        }

    }
    
    @classmethod
    def get_llm(cls, 
                model_name: str = "qwen-plus",
                temperature: float = 0,
                max_tokens: Optional[int] = None,
                dynamic_model: Optional[str] = None,
                **kwargs) -> ChatOpenAI:
        """
        获取指定模型的 ChatOpenAI 实例
        
        Args:
            model_name: 模型名称，支持预定义配置
            temperature: 温度参数，控制随机性
            max_tokens: 最大 token 数
            dynamic_model: 动态模型名称（用于支持动态模型的平台）
            **kwargs: 其他参数
            
        Returns:
            ChatOpenAI 实例
        """
        
        # 检查模型配置是否存在
        if model_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(cls.MODEL_CONFIGS.keys())}")
        
        # 获取模型配置
        config = cls.MODEL_CONFIGS[model_name].copy()
        
        # 检查 API Key
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError(f"未找到模型 {model_name} 的 API Key，请检查环境变量")
        
        # 处理动态模型
        if config.get("dynamic_model") and dynamic_model:
            # 动态模型平台，使用传入的模型名称
            actual_model = dynamic_model
        else:
            # 固定模型，使用配置中的模型名称
            actual_model = config.get("model", model_name)
        
        # 构建参数
        llm_params = {
            "model": actual_model,
            "openai_api_key": api_key,
            "openai_api_base": config["api_base"],
            "temperature": temperature,
            **kwargs
        }
        
        # 添加可选参数
        if max_tokens:
            llm_params["max_tokens"] = max_tokens
        
        return ChatOpenAI(**llm_params)
    
    @classmethod
    def get_default_llm(cls, **kwargs) -> ChatOpenAI:
        """获取默认的 LLM 实例（gpt-3.5-turbo）"""
        return cls.get_llm("qwen-plus", **kwargs)
    
    @classmethod
    def get_openai_llm(cls, model: str = "gpt-3.5-turbo", **kwargs) -> ChatOpenAI:
        """获取 OpenAI 官方模型"""
        return cls.get_llm(model, **kwargs)
    
    @classmethod
    def get_qwen_llm(cls, model: str = "qwen-plus", **kwargs) -> ChatOpenAI:
        """获取阿里云 Qwen 模型"""
        return cls.get_llm(model, **kwargs)
    
    @classmethod
    def get_kimi_llm(cls, **kwargs) -> ChatOpenAI:
        """获取月之暗面 Kimi 模型"""
        return cls.get_llm("kimi-k2", **kwargs)
    
    @classmethod
    def get_siliconflow_llm(cls, model: str, **kwargs) -> ChatOpenAI:
        """获取硅基流动模型（支持动态模型选择）"""
        return cls.get_llm("siliconflow", dynamic_model=model, **kwargs)
    
    @classmethod
    def get_dynamic_llm(cls, platform: str, model: str, **kwargs) -> ChatOpenAI:
        """获取动态模型（通用方法）"""
        return cls.get_llm(platform, dynamic_model=model, **kwargs)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """列出所有可用的模型配置"""
        return cls.MODEL_CONFIGS.copy()
    
    @classmethod
    def add_custom_model(cls, name: str, config: Dict[str, Any]):
        """添加自定义模型配置"""
        cls.MODEL_CONFIGS[name] = config
    



# 便捷函数
def get_llm(model_name: str = "qwen-plus", **kwargs) -> ChatOpenAI:
    """便捷函数：获取 LLM 实例"""
    return LLMFactory.get_llm(model_name, **kwargs)


def get_default_llm(**kwargs) -> ChatOpenAI:
    """便捷函数：获取默认 LLM 实例"""
    return LLMFactory.get_default_llm(**kwargs)


