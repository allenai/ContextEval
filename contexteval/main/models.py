import logging
import os
import time

import anthropic
import google.generativeai as genai
import openai
import tenacity
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
from openai import OpenAI
from together import Together

openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)


class GPT4:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        if model_name == "gpt-4":
            self.model_name = "gpt-4o"
        self.client = OpenAI(
            organization=os.getenv("OPENAI_ORGANIZATION"), api_key=os.getenv("OPENAI_API_KEY")
        )

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=1, max=60),
        stop=tenacity.stop_after_attempt(60),
    )
    def chat_completion_with_backoff(self, **kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def generate(self, input_text, max_len=512):  # gpt-4o-2024-05-13
        resp = self.chat_completion_with_backoff(
            model=self.model_name,
            messages=[{"role": "user", "content": input_text}],
            max_tokens=max_len,
        )
        output = resp.choices[0].message.content
        return output


class Gemini:
    def __init__(self, model_name="gemini-1.5-pro"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, input_text, max_len=512):
        orig_t = 5
        while True:
            try:
                response = self.model.generate_content(
                    input_text,
                    generation_config={"max_output_tokens": max_len},
                    safety_settings=[
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    ],
                )
                return response.text
            except Exception as e:
                print(e)
                if (
                    "Invalid operation: The `response.text` quick accessor requires the response to contain a valid `Part`"
                    in e.args[0]
                ):
                    return "No response. Finish Reason: Invalid operation."
                if "Invalid operation:" in e.args[0]:
                    return "No response. Finish Reason: Invalid operation."
                if isinstance(e, genai.types.generation_types.StopCandidateException):
                    print("No response. Finish Reason: Recitation.")
                    return "No response. Finish Reason: Recitation."
                orig_t *= 2
                time.sleep(orig_t)
                continue


class Claude:
    def __init__(self, model_name):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def generate(self, input_text, max_len=512):
        if self.model_name == "claude-3.5-sonnet":
            model = "claude-3-5-sonnet-20240620"
        elif self.model_name == "claude-3-haiku":
            model = "claude-3-haiku-20240307"
        elif self.model_name == "claude-3-opus":
            model = "claude-3-opus-20240229"
        while True:
            try:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=max_len,
                    messages=[{"role": "user", "content": input_text}],
                )
            except Exception as e:
                print(e)
                continue
            break
        return message.content[0].text


class TogetherAI:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"):
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.model_name = model_name

    def generate(self, input_text, max_len=512):
        tries = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": input_text}],
                    max_tokens=max_len,
                )
            except Exception as e:
                print("Exception:", e)
                tries += 1
                if tries == 3:
                    return "No response. Finish Reason: Timeout."
                else:
                    continue
            break

        return response.choices[0].message.content


class Jamba:
    def __init__(self, model_name="jamba-1.5-large"):
        self.model_name = model_name
        self.client = AI21Client(api_key=os.environ.get("AI21_API_KEY"))

    def generate(self, input_text, max_len=512):
        tries = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[ChatMessage(role="user", content=input_text)],
                    max_tokens=max_len,
                )

            except Exception as e:
                print(e)
                tries += 1
                if tries == 3:
                    return "No response. Finish Reason: Timeout."
                else:
                    continue
            break

        return response.choices[0].message.content
