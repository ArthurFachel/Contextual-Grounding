from typing import Any, Optional
import time
import json
import requests
import os
import re

from gas.providers.constant import CHOICE_PROMPT, GENERATE_PROMPT, TF_PROMPT
from dotenv import load_dotenv
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    pipeline,
    set_seed,
)
import os
from gas.logger import Logger
load_dotenv()
logging = Logger()
logger = logging.get_logger()


class AWSModel(DeepEvalBaseLLM):
    """
    Custom LLM wrapper for DeepEval.
    Integrado com lmformatenforcer para garantir outputs JSON válidos via API.
    """
    should_apply_chat_template = False

    def __init__(self,  model_params: dict[str, Any] = None,      # compatível com cli.py
        generation_params: dict[str, Any] = None,  # compatível com cli.py
        seed: Optional[int] = None,
        state: Optional[str] = None):

        model_params = model_params or {}
        generation_params = generation_params or {}
        self.state = state
        self.endpoint_url = model_params.get("endpoint_url") or os.getenv("AWS_ENDPOINT_URL")
        self.model = model_params.get("model") or os.getenv("AWS_GENERATE_MODEL")
        self.enhance_model = model_params.get("enhance_model") or os.getenv("AWS_ENHANCE_MODEL")
        
        self.models_list = model_params.get("models_list", [])
        self.baselines = model_params.get("baselines", [])


    def load_model(self):
        return None
    
    def enhance(self, prompt): 
        headers =   headers = {"Content-Type": "application/json"}
        
        input_match = re.search(r'### Input:\n(.*?)(?=\n### Response:|$)', prompt, re.DOTALL)
        input_only = input_match.group(1).strip() if input_match else prompt
        
        clean_prompt = f"{CHOICE_PROMPT} PROMPT:{input_only}"
        
        body = {
            "prompt": clean_prompt,
            "temperature": 0.1,
            "top_p": 0.9,
            "model": self.enhance_model,
            "max_tokens": 1048
        }

        while True:
            try:
                response = requests.post(
                    self.endpoint_url,
                    json=body,
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()
                
                data_raw = response.json()
                text_output = data_raw['body']['response']['output']['message']['content'][0]['text']
                
                return text_output

            except Exception as e:
                print(f"Erro na geração: {e}. Tentando novamente em 10s...")
                time.sleep(10)
                continue

   # def generate(self, prompt: str, schema: BaseModel = None) -> str | BaseModel:
        headers = {"Content-Type": "application/json"}
        print(f"prompt inicial: \n {prompt}")
        enhance_text = self.enhance(prompt)
        time.sleep(2)
        print(f"enhance_text: {enhance_text}")

        prompt = f"Added key words {enhance_text}, now answer: {prompt}"
        if schema:
            json_schema_str = json.dumps(schema.model_json_schema(), indent=2)
            prompt = f"{prompt}\n\nRespond ONLY with a JSON object that follows this schema:\n{json_schema_str}"
        print(prompt)
        bedrock_body = {
            "modelId": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "temperature": 0.1,
                "topP": 0.9,
                "maxTokens": 4096
            }
        }

        fallback_body = {
            "prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.9,
            "model": self.model,
            "max_tokens": 4096
        }

        retry_count = 0
        max_retries = 5
        use_bedrock_format = True

        while retry_count < max_retries:
            try:
                body = bedrock_body if use_bedrock_format else fallback_body
                
                logger.debug(f"Sending request with format: {'bedrock' if use_bedrock_format else 'fallback'}")
                
                response = requests.post(
                    self.endpoint_url,
                    json=body,
                    headers=headers,
                    timeout=60
                )

                try:
                    data = response.json()
                except json.JSONDecodeError:
                    raise RuntimeError(f"Invalid JSON response: {response.text}")

                status_code = response.status_code
                if "statusCode" in data:
                    status_code = data["statusCode"]
                
                error_msg = None
                if "error" in data:
                    error_msg = data.get("error", "")
                elif "body" in data and isinstance(data["body"], dict) and "error" in data["body"]:
                    error_msg = data["body"]["error"]
                elif "body" in data and isinstance(data["body"], str):
                    try:
                        body_parsed = json.loads(data["body"])
                        if "error" in body_parsed:
                            error_msg = body_parsed["error"]
                    except:
                        pass

                # Handle error responses
                if status_code >= 400:
                    if status_code == 429 or "Too many requests" in str(error_msg):
                        retry_count += 1
                        wait_time = 10 * (retry_count + 1)
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    elif status_code == 400 and error_msg and "Missing 'prompt' field" in error_msg:
                        if use_bedrock_format:
                            logger.info(f"Bedrock format not supported. Switching to fallback request format...")
                            use_bedrock_format = False
                            continue
                        else:
                            raise RuntimeError(f"Fallback format also failed: {error_msg}")
                    else:
                        raise RuntimeError(f"AWS API error ({status_code}): {error_msg or response.text}")

                if "body" in data and isinstance(data["body"], str):
                    try:
                        data["body"] = json.loads(data["body"])
                    except:
                        pass

                text_output = None
                
                if "output" in data:
                    text_output = data.get("output", {}).get("message", {}).get("content", [{}])[0].get("text")
                
                if not text_output and "body" in data and isinstance(data["body"], dict):
                    body_data = data["body"]
                    if "response" in body_data and "output" in body_data["response"]:
                        output = body_data["response"]["output"]
                        message = output.get("message", {})
                        content = message.get("content", [])
                        
                        if content and len(content) > 0:
                            # Try to find text in any content item (could be text or reasoning)
                            for content_item in content:
                                if "text" in content_item:
                                    text_output = content_item.get("text")
                                    if text_output:
                                        break
                                elif "reasoningContent" in content_item:
                                    reasoning = content_item["reasoningContent"]
                                    if isinstance(reasoning, dict):
                                        if "reasoningText" in reasoning:
                                            text_output = reasoning["reasoningText"].get("text", "")
                                        elif "text" in reasoning:
                                            text_output = reasoning.get("text", "")
                                    if text_output:
                                        break
                    
                    if not text_output:
                        text_output = (
                            body_data.get("response", {})
                                .get("output", {})
                                .get("message", {})
                                .get("content", [{}])[0]
                                .get("text")
                        )
                
                if not text_output and "text" in data:
                    text_output = data.get("text")

                if text_output:
                    return text_output.strip()
                else:
                    raise ValueError(f"Could not extract text from response: {data}")

            except requests.exceptions.Timeout:
                retry_count += 1
                logger.warning(f"Request timeout. Retrying {retry_count}/{max_retries} in 10s...")
                time.sleep(10)
            except requests.exceptions.ConnectionError as e:
                retry_count += 1
                logger.warning(f"Connection error: {e}. Retrying {retry_count}/{max_retries} in 10s...")
                time.sleep(10)
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error in generation: {e}. Retrying {retry_count}/{max_retries} in 10s...")
                time.sleep(10)

        raise RuntimeError(f"Failed to generate response after {max_retries} retries")
    
    async def a_generate(self, prompt: str, schema: BaseModel = None) -> str | BaseModel:
        """Async version - currently runs sync code in executor"""
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return  self.model

        from typing import Any, Optional

class DeepInfraModel(DeepEvalBaseLLM):
    """
    Custom LLM wrapper for DeepEval.
    Adapted for DeepInfra API using OpenAI-compatible endpoints.
    """
    should_apply_chat_template = False

    def __init__(
        self,  
        model_params: dict[str, Any] = None,      # compatível com cli.py
        generation_params: dict[str, Any] = None,  # compatível com cli.py
        seed: Optional[int] = None,
        state: Optional[str] = None
    ):
        model_params = model_params or {}
        generation_params = generation_params or {}
        self.state = state
        self.endpoint_url = model_params.get("endpoint_url") or os.getenv("DEEPINFRA_ENDPOINT_URL", "https://api.deepinfra.com/v1/openai/chat/completions")
        self.api_key = model_params.get("api_key") or os.getenv("DEEPINFRA_API_KEY")
        
        if not self.api_key:
            logger.warning("DeepInfra API key is missing. Ensure DEEPINFRA_API_KEY is set in your environment.")

        self.model = model_params.get("model") or os.getenv("DEEPINFRA_GENERATE_MODEL")
        self.enhance_model = model_params.get("enhance_model") or os.getenv("DEEPINFRA_ENHANCE_MODEL")
        


    def load_model(self):
        return None
    
    def enhance(self, prompt): 
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        input_match = re.search(r'### Input:\n(.*?)(?=\n### Response:|$)', prompt, re.DOTALL)
        input_only = input_match.group(1).strip() if input_match else prompt
        
        clean_prompt = f"{CHOICE_PROMPT} PROMPT:{input_only}"
        
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": clean_prompt
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1024
        }

        while True:
            try:
                response = requests.post(
                    self.endpoint_url,
                    json=body,
                    headers=headers,
                    timeout=180
                )
                response.raise_for_status()
                
                data_raw = response.json()
                text_output = data_raw['choices'][0]['message']['content']
                
                return text_output

            except Exception as e:
                print(f"Erro na geração (enhance): {e}. Tentando novamente em 10s...")
                time.sleep(10)
                continue

    def generate(self, prompt: str, schema: BaseModel = None) -> str | BaseModel:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        print(f"prompt inicial: \n {prompt}")
        enhance_text = self.enhance(prompt)
        time.sleep(10)
        
        if schema:
            json_schema_str = json.dumps(schema.model_json_schema(), indent=2)
            prompt = f"{prompt}\n\nRespond ONLY with a JSON object that follows this schema:\n{json_schema_str}"
            
        new_prompt = f"Added key words {enhance_text}, now answer: {prompt}"
        prompt = new_prompt
        print(prompt)
        
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4096
        }

        retry_count = 0
        max_retries = 5

        while retry_count < max_retries:
            try:
                logger.debug(f"Sending request to DeepInfra API for model {self.model}")
                
                response = requests.post(
                    self.endpoint_url,
                    json=body,
                    headers=headers,
                    timeout=180,
                    verify=False,
                )

                try:
                    data = response.json()
                except json.JSONDecodeError:
                    raise RuntimeError(f"Invalid JSON response: {response.text}")

                status_code = response.status_code
                
                # Handle error responses
                if status_code >= 400:
                    error_msg = data.get("error", {}).get("message", str(data))
                    
                    if status_code == 429 or "rate limit" in error_msg.lower():
                        retry_count += 1
                        wait_time = 10 * (retry_count + 1)
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"DeepInfra API error ({status_code}): {error_msg}")

                text_output = None
                
                if "choices" in data and len(data["choices"]) > 0:
                    text_output = data["choices"][0].get("message", {}).get("content")

                if text_output:
                    return text_output.strip()
                else:
                    raise ValueError(f"Could not extract text from response: {data}")

            except requests.exceptions.Timeout:
                retry_count += 1
                logger.warning(f"Request timeout. Retrying {retry_count}/{max_retries} in 10s...")
                time.sleep(10)
            except requests.exceptions.ConnectionError as e:
                retry_count += 1
                logger.warning(f"Connection error: {e}. Retrying {retry_count}/{max_retries} in 10s...")
                time.sleep(10)
            except Exception as e:
                if retry_count >= max_retries - 1:
                    retry_count += 1
                    continue
                retry_count += 1
                logger.warning(f"Error in generation: {e}. Retrying {retry_count}/{max_retries} in 10s...")
                time.sleep(10)
                if max_retries == 4:
                    print(f"Última tentativa falhou com erro: {e}. Não há mais retries disponíveis.")

        raise RuntimeError(f"Failed to generate response after {max_retries} retries")
    
    async def a_generate(self, prompt: str, schema: BaseModel = None) -> str | BaseModel:
        """Async version - currently runs sync code in executor"""
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model