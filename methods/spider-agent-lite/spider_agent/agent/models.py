"""
LLM Models Module

This module provides a unified interface for calling various Large Language Models (LLMs)
including OpenAI GPT, Claude, Grok, Gemini, and others. It handles API calls, retries,
error handling, and response formatting.
"""

import logging
import os
import time
from openai import AzureOpenAI
from groq import Groq
import google.generativeai as genai
import requests
from openai import OpenAI

logger = logging.getLogger("api-llms")


def call_llm(payload):
    """
    Call a Large Language Model with the given payload.

    This function routes the request to the appropriate API based on the model name
    and handles retries, error handling, and response formatting.

    Args:
        payload (dict): Request payload containing model, messages, and parameters.

    Returns:
        tuple: (success: bool, response: str or error_code)
    """
    model = payload["model"]
    stop = ["Observation:", "\n\n\n\n", "\n \n \n"]

    # Handle OpenAI GPT models
    # Handle OpenAI GPT models
    if model.startswith("gpt") or model.startswith("chatgpt-4o-latest"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        }
        logger.info("Generating content with GPT model: %s", model)

        # Retry up to 3 times with exponential backoff
        for i in range(3):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                output_message = response.json()["choices"][0]["message"]["content"]
                return True, output_message
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                if hasattr(e, "response") and e.response is not None:
                    error_info = e.response.json()
                    code_value = error_info["error"]["code"]
                    # Handle content filter by adding disclaimer
                    if code_value == "content_filter":
                        if not payload["messages"][-1]["content"][0]["text"].endswith(
                            "They do not represent any real events or entities. ]"
                        ):
                            payload["messages"][-1]["content"][0][
                                "text"
                            ] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value
                else:
                    code_value = "unknown_error"
                logger.error("Retrying ...")
                time.sleep(4 * (2 ** (i + 1)))
        return False, code_value

    # Handle OpenAI O1 models (o1, o3, o4-mini)
    elif (
        model.startswith("o1") or model.startswith("o3") or model.startswith("o4-mini")
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        }
        logger.info("Generating content with GPT model: %s", model)

        messages = payload["messages"]
        temperature = payload["temperature"]

        # Convert messages to O1 format (system messages become user messages)
        o1_messages = []
        for i, message in enumerate(messages):
            o1_message = {
                "role": message["role"] if message["role"] != "system" else "user",
                "content": "",
            }
            for part in message["content"]:
                o1_message["content"] = part["text"] if part["type"] == "text" else ""

            o1_messages.append(o1_message)

        payload["messages"] = o1_messages
        payload["max_completion_tokens"] = 10000
        # Remove unsupported parameters for O1 models
        del payload["max_tokens"]
        del payload["temperature"]
        if "top_p" in payload:
            del payload["top_p"]

        # Retry with exponential backoff
        for i in range(3):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )

                response_json = response.json()
                if "choices" in response_json and response_json["choices"]:
                    output_message = response_json["choices"][0]["message"]["content"]
                    return True, output_message
                else:
                    logger.error(
                        f"OpenAI API returned unexpected response: {response_json}"
                    )
                    return False, response_json.get("error", {}).get(
                        "message", "No 'choices' in response"
                    )
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value

    # Handle Azure OpenAI models
    elif model.startswith("azure"):
        client = AzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            api_version="2024-02-15-preview",
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
        )
        model_name = model.split("/")[-1]
        # Retry up to 3 times
        for i in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=payload["messages"],
                    max_tokens=payload["max_tokens"],
                    top_p=payload["top_p"],
                    temperature=payload["temperature"],
                    stop=stop,
                )
                response = response.choices[0].message.content
                return True, response
            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                error_info = e.response.json()
                code_value = error_info["error"]["code"]
                # Handle content filter
                if code_value == "content_filter":
                    if not payload["messages"][-1]["content"][0]["text"].endswith(
                        "They do not represent any real events or entities. ]"
                    ):
                        payload["messages"][-1]["content"][0][
                            "text"
                        ] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                if code_value == "context_length_exceeded":
                    return False, code_value
                logger.error("Retrying ...")
                time.sleep(10 * (2 ** (i + 1)))
        return False, code_value

    # Handle Claude models
    elif model.startswith("claude"):
        messages = payload["messages"]
        max_tokens = payload.get("max_tokens", 4096)
        temperature = payload.get("temperature", 1.0)
        top_p = payload.get("top_p", 0.7)

        # Convert to Claude's message format
        claude_messages = []

        # Handle system message if present
        system_content = None
        if messages and messages[0]["role"] == "system":
            system_content = ""
            for part in messages[0]["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    system_content += part["text"]
                elif isinstance(part, str):
                    system_content += part
            messages = messages[1:]  # Remove system message from regular messages

        # Process regular messages
        for message in messages:
            role = message["role"]
            message_content = []

            # Handle content which might be a list of parts or a string
            content = message["content"]
            if isinstance(content, str):
                # Convert string content to text part
                message_content.append({"type": "text", "text": content})
            else:
                # Process list of content parts
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            message_content.append(
                                {"type": "text", "text": part["text"]}
                            )
                        elif part.get("type") == "image_url":
                            # Handle image data - convert to Claude's image format
                            image_url = part["image_url"]["url"]
                            if image_url.startswith("data:image/"):
                                # Extract base64 data
                                media_type = image_url.split(";")[0].replace(
                                    "data:", ""
                                )
                                base64_data = image_url.split(",")[1]

                                message_content.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": base64_data,
                                        },
                                    }
                                )
                    elif isinstance(part, str):
                        # Handle plain string content
                        message_content.append({"type": "text", "text": part})

            claude_messages.append({"role": role, "content": message_content})

        # Prepare the request payload for Claude API
        claude_payload = {
            "model": model,
            "messages": claude_messages,
            "max_tokens": max_tokens,
            # "temperature": temperature,
            # "top_p": top_p
        }

        # Add system prompt if present
        if system_content:
            claude_payload["system"] = system_content

        headers = {
            "Accept": "application/json",
            "Anthropic-Version": "messages-2023-06-01",
            "Anthropic-Beta": "messages-2023-12-15",
            "Authorization": f'Bearer {os.environ["ANTHROPIC_API_KEY"]}',
            "Content-Type": "application/json",
        }

        # Make the API request with retries
        for i in range(3):
            try:
                logger.info(f"Calling Claude API with model: {model}")
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=claude_payload,
                )

                if response.status_code == 200:
                    response_data = response.json()
                    return True, response_data["content"][0]["text"]
                else:
                    error_info = response.json()
                    logger.error(f"Claude API error: {error_info}")

                    # Handle content filter errors
                    if (
                        "error" in error_info
                        and error_info["error"].get("type")
                        == "content_policy_violation"
                    ):
                        # Add a disclaimer to the last message
                        last_message = claude_payload["messages"][-1]
                        for content_part in last_message["content"]:
                            if content_part["type"] == "text":
                                if not content_part["text"].endswith(
                                    "They do not represent any real events or entities. ]"
                                ):
                                    content_part[
                                        "text"
                                    ] += " [ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                                break

                    # Handle context length errors
                    if (
                        "error" in error_info
                        and error_info["error"].get("type") == "context_length_exceeded"
                    ):
                        return False, "context_length_exceeded"

                    # Retry with exponential backoff
                    logger.warning(f"Retrying Claude API call ({i+1}/3)...")
                    time.sleep(10 * (2 ** (i + 1)))

            except Exception as e:
                logger.error(f"Failed to call Claude API: {str(e)}")
                time.sleep(10 * (2 ** (i + 1)))

        return False, "api_call_failed"

    # Handle Mistral models via Groq
    elif model.startswith("mistral"):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        top_p = payload["top_p"]
        temperature = payload["temperature"]

        # Convert messages to simple format
        mistral_messages = []
        for i, message in enumerate(messages):
            mistral_message = {"role": message["role"], "content": ""}
            for part in message["content"]:
                mistral_message["content"] = (
                    part["text"] if part["type"] == "text" else ""
                )
            mistral_messages.append(mistral_message)

        # Initialize Groq client
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        print(client)
        # Retry up to 2 times
        for i in range(2):
            try:
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=mistral_messages,
                    model=model,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    stop=stop,
                )
                return True, response.choices[0].message.content

            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))
                if hasattr(e, "response"):
                    error_info = e.response.json()
                    code_value = error_info["error"]["code"]
                    # Handle content filter
                    if code_value == "content_filter":
                        if not payload["messages"][-1]["content"][0]["text"].endswith(
                            "They do not represent any real events or entities. ]"
                        ):
                            payload["messages"][-1]["content"][0][
                                "text"
                            ] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                    if code_value == "context_length_exceeded":
                        return False, code_value
                else:
                    code_value = ""
                logger.error("Retrying ...")

        return False, code_value

    # Handle models via Together AI (Qwen, DeepSeek, etc.)
    elif model in [
        "qwen_api_32b-instruct-fp16",
        "qwen_api_2_5_72b",
        "llamaapi_3.3",
        "deepSeek-R1",
        "deepSeek-v3",
        "Qwen3",
    ]:
        model_mapping = {
            "qwen_api_32b-instruct-fp16": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "qwen_api_2_5_72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "llamaapi_3.3": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "deepSeek-R1": "deepseek-ai/DeepSeek-R1",
            "deepSeek-v3": "deepseek-ai/DeepSeek-V3",
            "Qwen3": "Qwen/Qwen3-235B-A22B-fp8-tput",
        }
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        temperature = payload["temperature"]

        # Convert messages to simple format
        mistral_messages = []
        for i, message in enumerate(messages):
            mistral_message = {"role": message["role"], "content": ""}
            for part in message["content"]:
                mistral_message["content"] = (
                    part["text"] if part["type"] == "text" else ""
                )
            mistral_messages.append(mistral_message)

        # Initialize Together client
        from together import Together

        api_key = os.getenv("TOGETHER_API_KEY")  # 读取环境变量中的 API Key
        if not api_key:
            raise ValueError(
                "API key not found. Please set TOGETHER_API_KEY as an environment variable."
            )
        client = Together()

        # Retry up to 3 times
        for i in range(3):
            try:
                logger.info("Generating content with model: %s", model)
                response = client.chat.completions.create(
                    messages=mistral_messages,
                    model=model_mapping[model],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return True, response.choices[0].message.content

            except Exception as e:
                logger.error("Failed to call LLM: " + str(e))
                time.sleep(10 * (2 ** (i + 1)))

                code_value = "context_length_exceeded"
                logger.error("Retrying ...")

        return False, code_value

    # Handle Grok models
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
                if part["type"] == "text":
                    content += part["text"]
            grok_messages.append({"role": message["role"], "content": content})

        # Check for API key
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            logger.error("GROK_API_KEY not set in environment variables.")
            return False, "missing_api_key"

        # Retry up to 3 times
        for i in range(3):
            try:
                logger.info("Generating content with Grok model: %s", model)
                client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                response = client.chat.completions.create(
                    model=model,
                    messages=grok_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                return True, response.choices[0].message.content
            except Exception as e:
                logger.error("Failed to call Grok API: %s", str(e))
                time.sleep(5 * (2 ** (i + 1)))
                code_value = "unknown_error"
        return False, code_value

    # Handle Gemini models
    elif model.startswith("gemini-2.5-pro-preview-03-25") or model.startswith(
        "gemini-2.5-pro-preview-05-06"
    ):
        messages = payload["messages"]
        max_tokens = payload["max_tokens"]
        temperature = payload["temperature"]

        # Initialize Gemini client
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_model = genai.GenerativeModel(
            model_name="models/gemini-2.5-pro-preview-03-25"
        )

        # Convert messages to Gemini format
        gemini_messages = []
        for message in messages:
            content_parts = []
            for part in message["content"]:
                if part["type"] == "text":
                    content_parts.append(part["text"])
                elif part["type"] == "image_url":
                    # Handle image if needed (Gemini supports multimodal input)
                    image_data = part["image_url"]["url"].replace(
                        "data:image/png;base64,", ""
                    )
                    content_parts.append({"mime_type": "image/png", "data": image_data})

            # Gemini doesn't use explicit role separation in the same way; combine content
            if message["role"] == "system":
                # Prepend system message as text if needed
                gemini_messages.insert(0, "\n".join(content_parts))
            else:
                gemini_messages.append(
                    "\n".join(
                        content_parts
                        if isinstance(content_parts, list)
                        else [content_parts]
                    )
                )

        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "stop_sequences": stop,
        }

        # Retry up to 3 times
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
                # Handle content filter
                if "content" in str(e).lower():
                    if not messages[-1]["content"][0]["text"].endswith(
                        "They do not represent any real events or entities. ]"
                    ):
                        messages[-1]["content"][0][
                            "text"
                        ] += "[ Note: The data and code snippets are purely fictional and used for testing and demonstration purposes only. They do not represent any real events or entities. ]"
                elif "length" in str(e).lower():
                    return False, "context_length_exceeded"
        return False, code_value
