from pprint import pprint

from imagine import ImagineClient, ModelType
from imagine import ChatMessage

client = ImagineClient(api_key="7256efb0-e4e9-4be9-b7c3-f79073192f85", endpoint="https://aisuite.cirrascale.com/apis/v2")

all_models = client.get_available_models_by_type()
pprint(all_models)

llm_models = client.get_available_models(model_type=ModelType.LLM)
pprint(llm_models)

chat_response = client.chat(
    messages=[ChatMessage(role="user", content="What is the best Spanish cheese?")],
    model="Llama-3.1-8B",
)

print(chat_response.first_content)