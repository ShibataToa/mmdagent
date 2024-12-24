import sys
import json
import time
import re
import string

import openai
from queue import Queue
from base import MMDAgentEXLabel
import google.generativeai as genai


class ResponseGenerator:
    def __init__(self, config, asr_timestamp, query, dialogue_history, prompts):

        # 設定の読み込み
        self.max_tokens = config["ChatGPT"]["max_tokens"]
        self.max_message_num_in_context = config["ChatGPT"][
            "max_message_num_in_context"
        ]
        self.model = config["ChatGPT"]["response_generation_model"]

        # 処理対象のユーザ発話に関する情報
        self.asr_timestamp = asr_timestamp
        self.query = query
        self.dialogue_history = dialogue_history
        self.prompts = prompts

        # 生成中の応答を保持・パースする変数
        self.response_fragment = ""
        self.punction_pattern = re.compile("[ 、 。 ! ?]")
        # ChatGPTに入力する対話文脈
        messages = []

        # 過去の対話履歴を対話文脈に追加
        i = max(0, len(self.dialogue_history) - self.max_message_num_in_context)
        messages.extend(self.dialogue_history[i:])

        # プロンプトおよび新しいユーザ発話を対話文脈に追加
        if query:
            messages.extend(
                [
                    {"role": "user", "content": self.prompts["RESP"]},
                    {"role": "system", "content": "OK"},
                    {"role": "user", "content": query},
                ]
            )
        # 新しいユーザ発話が存在せず自ら発話する場合のプロンプトを対話文脈に追加
        else:
            messages.extend([{"role": "user", "content": prompts["TO"]}])

        self.log(f"Call ChatGPT: {query=}")

        # ChatGPTに対話文脈を入力してストリーミング形式で応答の生成を開始
        self.response = openai.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens, stream=True
        )

    # Dialogueのsend_response関数で呼び出され，応答の断片を順次返す
    def __next__(self):
        # 引数（例: '1_喜び,6_会釈'）をパースして，expressionとactionを取得
        def _parse_split(split):
            expression = MMDAgentEXLabel.id2expression[0]
            action = MMDAgentEXLabel.id2action[0]

            # expression/actionを取得
            if "," in split:
                expression, action = split.split(",", 1)

                expression = expression.split("_")[0]
                expression = int(expression) if expression.isdigit() else 0
                expression = MMDAgentEXLabel.id2expression[expression]

                action = action.split("_")[0]
                action = int(action) if action.isdigit() else 0
                action = MMDAgentEXLabel.id2action[action]

            return {"expression": expression, "action": action}

        # ChatGPTの応答を順次パースして返す
        for chunk in self.response:
            # chunk.choices は属性としてアクセス
            chunk_message = chunk.choices[0].delta
            if hasattr(chunk_message, "content") and chunk_message.content:
                new_token = chunk_message.content
                self.response_fragment += new_token
                splits = self.punction_pattern.split(self.response_fragment, 1)
                if len(splits) == 2:
                    result = splits[0]
                    self.response_fragment = splits[1]
                    return {"phrase": result}
        raise StopIteration

    # ResponseGeneratorをイテレータ化
    def __iter__(self):
        return self

    # デバッグ用のログ出力
    def log(self, *args, **kwargs):
        print(f"[{time.time():.5f}]", *args, flush=True, **kwargs)


class ResponseChatGPT:
    def __init__(self, config, prompts):
        self.config = config
        self.prompts = prompts

        # 設定の読み込み
        openai.api_key = config["ChatGPT"]["api_key"]

        # 入力されたユーザ発話に関する情報を保持する変数
        self.user_utterance = ""
        self.response = ""
        self.last_asr_iu_id = ""
        self.asr_time = 0.0

    # ChatGPTの呼び出しを開始
    def run(
        self,
        asr_timestamp,
        user_utterance,
        dialogue_history,
        last_asr_iu_id,
        parent_llm_buffer,
    ):
        self.user_utterance = user_utterance
        self.last_asr_iu_id = last_asr_iu_id
        self.asr_time = asr_timestamp

        # ChataGPTを呼び出して応答の生成を開始
        self.response = ResponseGenerator(
            self.config, asr_timestamp, user_utterance, dialogue_history, self.prompts
        )

        # 自身をDialogueモジュールが持つLLMバッファに追加
        parent_llm_buffer.put(self)


if __name__ == "__main__":

    openai.api_key = ""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello!"}]
        )
        print("API test success:", response)
    except openai.error.AuthenticationError as e:
        print("Authentication Error:", e)

    config = {
        "ChatGPT": {
            "max_tokens": 64,
            "max_message_num_in_context": 3,
            "response_generation_model": "gpt-3.5-turbo",
        }
    }

    asr_timestamp = time.time()
    query = "今日は良い天気だね"
    dialogue_history = []
    prompts = {}

    with open("./prompt/response.txt") as f:
        prompts["RESP"] = f.read()

    response_generator = ResponseGenerator(
        config, asr_timestamp, query, dialogue_history, prompts
    )

    for part in response_generator:
        response_generator.log(part)


"""
class GeminiResponseWrapper:
    def __init__(self, response):
        self.response = response
        self.rest_response = Queue()
        self.last_gen = False
        self.done = False

    def __next__(self):
        while True:
            if not self.rest_response.empty():
                return self.gen()
            if self.done:
                break

            for chunk in self.response:
                for c in chunk.text:
                    self.rest_response.put(c)
                break
            else:
                self.done = True

        if not self.last_gen:
            self.last_gen = True
            return {"choices": [{"delta": {}}]}
        raise StopIteration

    def gen(self):
        c = self.rest_response.get()
        return {"choices": [{"delta": {"content": c}}]}

    def __iter__(self):
        return self

"""