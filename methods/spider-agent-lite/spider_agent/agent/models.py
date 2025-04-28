import base64
import json
import logging
import os
import re
import time
from http import HTTPStatus
from io import BytesIO

from openai import AzureOpenAI
from typing import Dict, List, Optional, Tuple, Any, TypedDict
import dashscope
from groq import Groq
import google.generativeai as genai
import openai
import requests
import tiktoken
import signal
from openai import OpenAI

logger = logging.getLogger("api-llms")


def call_llm(payload):
    model = payload["model"]
    stop = ["Observation:","\n\n\n\n","\n \n \n"]
    if model.startswith("gpt") or model.startswith("chatgpt-4o-latest"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        logger.info("Generating content with GPT model: %s", model)
        

        for i in range(3):
            try:
                response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                output_message = response.json()['choices'][0]['message']['content']
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                if hasattr(e, 'response') and e.response is not None:
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = 'unknown_error'
                logger.error("Retrying ...")
                time.sleep(4 * (2 ** (i + 1)))
        return False, code_value
    
    elif model.startswith("o1") or model.startswith("o3"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        logger.info("Generating content with GPT model: %s", model)
        
        messages = payload["messages"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]
            
            
        o1_messages = []

        for i, message in enumerate(messages):
            o1_message = {
                "role": message["role"] if message["role"] != "system" else "user",
                "content": ""
            }
            for part in message["content"]:
                o1_message['content'] = part['text'] if part['type'] == "text" else ""
                
                o1_messages.append(o1_message)

        payload["messages"] = o1_messages
        payload["max_completion_tokens"] = 10000
        del payload['max_tokens']
        del payload["temperature"]
        del payload["top_p"]

        for i in range(3):
            try:
                response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                output_message = response.json()['choices'][0]['message']['content']
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value

    elif model.startswith("azure"):
        client = AzureOpenAI(
            api_key = os.environ['AZURE_API_KEY'],  
            api_version = "2024-02-15-preview",
            azure_endpoint = os.environ['AZURE_ENDPOINT']
            )
        model_name = model.split("/")[-1]
        for i in range(3):
            try:
                response = client.chat.completions.create(model=model_name,messages=payload['messages'], max_tokens=payload['max_tokens'], top_p=payload['top_p'], temperature=payload['temperature'], stop=stop)
                response = response.choices[0].message.content
                # logger.info(f"Input: \n{payload['messages']}\nOutput:{response}")
                return True, response
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                error_info = e.response.json()  
                code_value = error_info['error']['code']
                if code_value == "content_filter":
                    if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                        payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                if code_value == "context_length_exceeded":
                    return False, code_value        
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value
        
    elif model.startswith("claude"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        gemini_messages = []

        for i, message in enumerate(messages):
            gemini_message = {
                "role": message["role"],
                "content": []
            }
            assert len(message["content"]) in [1, 2], "One text, or one text with one image"
            for part in message["content"]:

                if part['type'] == "image_url":
                    image_source = {}
                    image_source["type"] = "base64"
                    image_source["media_type"] = "image/png"
                    image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
                    gemini_message['content'].append({"type": "image", "source": image_source})

                if part['type'] == "text":
                    gemini_message['content'].append({"type": "text", "text": part['text']})

            gemini_messages.append(gemini_message)
        
        if gemini_messages[0]['role'] == "system":
            gemini_system_message_item = gemini_messages[0]['content'][0]
            gemini_messages[1]['content'].insert(0, gemini_system_message_item)
            gemini_messages.pop(0)


        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {os.environ["GEMINI_API_KEY"]}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }  
        
        payload = json.dumps({"model": model,"messages": gemini_messages,"max_tokens": max_tokens,"temperature": temperature,"top_p": top_p})


        
        for i in range(3):
            try:
                response = requests.request("POST", "https://api2.aigcbest.top/v1/chat/completions", headers=headers, data=payload)
                logger.info(f"response_code {response.status_code}")
                if response.status_code == 200:
                    return True, response.json()['choices'][0]['message']['content']
                else:
                    error_info = response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value
                    logger.error("Retrying ...")
                    time.sleep(10 * (2 ** (i + 1)))
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                code_value = "context_length_exceeded"
        return False, code_value
                           

    elif model.startswith("mistral"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        mistral_messages = []

        for i, message in enumerate(messages):
            mistral_message = {
                "role": message["role"],
                "content": ""
            }

            for part in message["content"]:
                mistral_message['content'] = part['text'] if part['type'] == "text" else ""

            mistral_messages.append(mistral_message)

        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        print(client)
        for i in range(2):
            try:
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=mistral_messages,
                    model=model,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    stop = stop
                )
                return True, response.choices[0].message.content
                
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                if hasattr(e, 'response'):
                    error_info = e.response.json()  
                    code_value = error_info['error']['code']
                    if code_value == "content_filter":
                        if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                            payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value        
                else:
                    code_value = ""
                logger.error("Retrying ...") 

        return False, code_value
        
    # elif model.startswith("llama"):
    #     messages = payload["messages"]
    #     max_tokens = payload["max_tokens"]
    #     top_p = payload["top_p"]
    #     temperature = payload["temperature"]

    #     llama_messages = []

    #     for i, message in enumerate(messages):
    #         llama_message = {
    #             "role": message["role"],
    #             "content": ""
    #         }
    #         for part in message["content"]:
    #             llama_message['content'] = part['text'] if part['type'] == "text" else ""
    #             llama_messages.append(llama_message)   
    #         llama_messages.append(llama_message)

    #     for i in range(3):
    #         try:
    #             logger.info("Generating content with model: %s", model)
    #             response = dashscope.Generation.call(
    #                 model=model,
    #                 messages=llama_messages,
    #                 result_format="message",
    #                 max_length=max_tokens,
    #                 top_p=top_p,
    #                 temperature=temperature,
    #                 stop = stop
    #             )
    #             import pdb; pdb.set_trace()
    #             return True, response['output']['choices'][0]['message']['content']

    #         except Exception as e:
    #             logger.error("Failed to call LLM: " + str(e))
    #             time.sleep(4 * (2 ** (i + 1)))
    #             if hasattr(e, 'response'):
    #                 error_info = e.response.json()  
    #                 code_value = error_info['error']['code']
    #                 if code_value == "content_filter":
    #                     if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
    #                         payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
    #             else:
    #                 code_value = "context_length_exceeded"
    #             logger.error("Retrying ...")
    #     return False, code_value
    
    
    
    # elif model.startswith("deepseek"):
        
    #     messages = payload["messages"]
    #     max_tokens = payload["max_tokens"]
    #     top_p = payload["top_p"]
    #     temperature = payload["temperature"]
    #     from openai import OpenAI
        
    #     deepseek_messages = []

    #     for i, message in enumerate(messages):
    #         deepseek_message = {
    #             "role": message["role"],
    #             "content": ""
    #         }
    #         for part in message["content"]:
    #             deepseek_message['content'] = part['text'] if part['type'] == "text" else ""
    #             deepseek_messages.append(deepseek_message)
    #     client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    #     for i in range(3):
    #         try:
    #             response = client.chat.completions.create(
    #                 model=model,
    #                 messages=deepseek_messages,
    #                 max_tokens=max_tokens,
    #                 top_p=top_p,
    #                 temperature=temperature
    #             )
    #             output_message = json.loads(response.json())['choices'][0]['message']['content']
    #             return True, output_message
    #         except Exception as e:
    #             logger.error("Failed to call LLM: " + str(e))
    #             if hasattr(e, 'response') and e.response is not None:
    #                 error_info = e.response.json()  
    #                 code_value = error_info['error']['code']

    #                 if code_value == "content_filter":
    #                     last_message = messages[-1]
    #                     if 'content' in last_message and isinstance(last_message['content'], str):
    #                         if not last_message['content'].endswith("They do not represent any real events or entities. ]"):
    #                             last_message['content'] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
    #                     else:
    #                         logger.error("Unexpected message structure in 'messages'. Skipping content modification.")
    #                 elif code_value == "context_length_exceeded":
    #                     return False, code_value        
    #             else:
    #                 code_value = 'unknown_error'
                
    #             logger.error("Retrying ...")
    #             time.sleep(10 * (2 ** (i + 1)))
    #         return False, code_value


        
    # elif model.startswith("qwen") or model.startswith("Qwen") or model.startswith("llama3.1"):
    #     messages = payload["messages"]
    #     max_tokens = payload["max_tokens"]
    #     top_p = payload["top_p"]
    #     temperature = payload["temperature"]
        
    #     if model.startswith("llama3.1"):
    #         max_tokens = 2000

    #     qwen_messages = []

    #     for i, message in enumerate(messages):
    #         qwen_message = {
    #             "role": message["role"],
    #             "content": []
    #         }
    #         assert len(message["content"]) in [1, 2], "One text, or one text with one image"
    #         for part in message["content"]:
    #             qwen_message['content'].append({"text": part['text']}) if part['type'] == "text" else None

    #         qwen_messages.append(qwen_message)

    #     for i in range(3):
    #         try:
    #             logger.info("Generating content with model: %s", model)
    #             response = dashscope.Generation.call(
    #                 model=model,
    #                 messages=qwen_messages,
    #                 result_format="message",
    #                 max_length=max_tokens,
    #                 top_p=top_p,
    #                 temperature=temperature,
    #                 stop = stop
    #             )
    #             return True, response['output']['choices'][0]['message']['content']

    #         except Exception as e:
    #             logger.error("Failed to call LLM: " + str(e))
    #             time.sleep(3 * (2 ** (i + 1)))
    #             if hasattr(e, 'response'):
    #                 error_info = e.response.json()  
    #                 code_value = error_info['error']['code']
    #                 if code_value == "content_filter":
    #                     if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
    #                         payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
    #             else:
    #                 code_value = "context_length_exceeded"
    #             logger.error("Retrying ...")
    #     return False, code_value


        
    # elif model.startswith("codellama") or model.startswith("mistralai"):
    elif model in ["qwen_api_32b-instruct-fp16", "qwen_api_2_5_72b", "llamaapi_3.3","deepSeek-R1"]:
        model_mapping = {
            "qwen_api_32b-instruct-fp16": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "qwen_api_2_5_72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "llamaapi_3.3": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "deepSeek-R1":"deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        }
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        # if model == "codellama/CodeLlama-70b-Instruct-hf":
        #     max_tokens = 800
        # top_p = payload["top_p"]
        temperature = payload["temperature"]

        mistral_messages = []

        for i, message in enumerate(messages):
            mistral_message = {
                "role": message["role"],
                "content": ""
            }

            for part in message["content"]:
                mistral_message['content'] = part['text'] if part['type'] == "text" else ""

            mistral_messages.append(mistral_message)
        from together import Together

        api_key = os.getenv("TOGETHER_API_KEY")  # 读取环境变量中的 API Key
        if not api_key:
            raise ValueError("API key not found. Please set TOGETHER_API_KEY as an environment variable.")
        client = Together()
        # total_tokens = len(mistral_messages) + max_tokens
        # if total_tokens > 8193:
        #     max_tokens = 8193 - len(mistral_messages)
        # print(f"Total tokens: {total_tokens}, Max new tokens: {max_tokens}")
        for i in range(3):
            try:
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=mistral_messages,
                    model=model_mapping[model],
                    max_tokens=max_tokens,
                    # top_p=top_p,
                    temperature=temperature
                )
                return True, response.choices[0].message.content
                
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                # if hasattr(e, 'response'):
                #     error_info = e.response  
                #     code_value = error_info['error']['param']
                #     if "content" in code_value:
                #         if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                #             payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                #     if code_value == "max_tokens":
                #         return False, code_value        
                # else:
                code_value = "context_length_exceeded"
                logger.error("Retrying ...")

        return False, code_value



    elif model.startswith("grok"):
        messages = payload["messages"]
        max_tokens = payload.get("max_tokens", 256)
        temperature = payload.get("temperature", 0.5)
        stop = payload.get("stop", None)

        # Prepare messages for OpenAI API format
        grok_messages = []
        for message in messages:
            content = ""
            for part in message["content"]:
                if part['type'] == "text":
                    content += part['text']
            grok_messages.append({
                "role": message["role"],
                "content": content
            })

        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            logger.error("GROK_API_KEY not set in environment variables.")
            return False, "missing_api_key"

        for i in range(3):
            try:
                logger.info("Generating content with Grok model: %s", model)
                client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                response = client.chat.completions.create(
                    model=model,
                    messages=grok_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop
                )
                # The SDK returns an object, not a requests.Response
                return True, response.choices[0].message.content
            except Exception as e:
                logger.error("Failed to call Grok API: %s", str(e))
                time.sleep(5 * (2 ** (i + 1)))
                code_value = "unknown_error"
        return False, code_value

    elif model.startswith("gemini-2.5-pro-preview-03-25"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        temperature = payload["temperature"]

        # Initialize Gemini client
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-pro-preview-03-25")

        # Convert messages to Gemini format
        gemini_messages = []
        for message in messages:
            content_parts = []
            for part in message["content"]:
                if part['type'] == "text":
                    content_parts.append(part['text'])
                elif part['type'] == "image_url":
                    # Handle image if needed (Gemini supports multimodal input)
                    image_data = part['image_url']['url'].replace("data:image/png;base64,", "")
                    content_parts.append({
                        "mime_type": "image/png",
                        "data": image_data
                    })

            # Gemini doesn't use explicit role separation in the same way; combine content
            if message["role"] == "system":
                # Prepend system message as text if needed
                gemini_messages.insert(0, "\n".join(content_parts))
            else:
                gemini_messages.append("\n".join(content_parts if isinstance(content_parts, list) else [content_parts]))

        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "stop_sequences": stop
        }

        for i in range(3):
            try:
                logger.info("Generating content with Gemini model: %s", model)
                # Use generate_content for Gemini API
                response = gemini_model.generate_content(
                    gemini_messages,
                    # generation_config=generation_config
                )
                print(response.text)
                return True, response.text
            except Exception as e:
                logger.error("Failed to call Gemini API: " + str(e))
                time.sleep(5 * (2 ** (i + 1)))
                code_value = "gemini_error"
                if "content" in str(e).lower():
                    if not messages[-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
                        messages[-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                elif "length" in str(e).lower():
                    return False, "context_length_exceeded"
        return False, code_value
    # elif model == "gemini-1.5-pro-latest":
    #     messages = payload["messages"]
    #     max_tokens = payload["max_tokens"]
    #     top_p = payload["top_p"]
    #     temperature = payload["temperature"]

    #     gemini_messages = []

    #     for i, message in enumerate(messages):
    #         gemini_message = {
    #             "role": message["role"],
    #             "content": []
    #         }
    #         assert len(message["content"]) in [1, 2], "One text, or one text with one image"
    #         for part in message["content"]:

    #             if part['type'] == "image_url":
    #                 image_source = {}
    #                 image_source["type"] = "base64"
    #                 image_source["media_type"] = "image/png"
    #                 image_source["data"] = part['image_url']['url'].replace("data:image/png;base64,", "")
    #                 gemini_message['content'].append({"type": "image", "source": image_source})

    #             if part['type'] == "text":
    #                 gemini_message['content'].append({"type": "text", "text": part['text']})

    #         gemini_messages.append(gemini_message)

    #     headers = {
    #         'Accept': 'application/json',
    #         'Authorization': f'Bearer {os.environ["GEMINI_API_KEY"]}',
    #         'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    #         'Content-Type': 'application/json'
    #     }  
        
    #     payload = json.dumps({"model": model,"messages": gemini_messages,"max_tokens": max_tokens,"temperature": temperature,"top_p": top_p})


        
    #     for i in range(3):
    #         try:
    #             response = requests.request("POST", "https://api2.aigcbest.top/v1/chat/completions", headers=headers, data=payload)
    #             logger.info(f"response_code {response.status_code}")
    #             if response.status_code == 200:
    #                 return True, response.json()['choices'][0]['message']['content']
    #             else:
    #                 error_info = response.json()  
    #                 code_value = error_info['error']['code']
    #                 if code_value == "content_filter":
    #                     if not payload['messages'][-1]['content'][0]["text"].endswith("They do not represent any real events or entities. ]"):
    #                         payload['messages'][-1]['content'][0]["text"] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
    #                 if code_value == "context_length_exceeded":
    #                     return False, code_value
    #                 logger.error("Retrying ...")
    #                 time.sleep(10 * (2 ** (i + 1)))
    #         except Exception as e:
    #             logger.error("Failed to call LLM: " + str(e))
    #             time.sleep(10 * (2 ** (i + 1)))
    #             code_value = "context_length_exceeded"
    #     return False, code_value
