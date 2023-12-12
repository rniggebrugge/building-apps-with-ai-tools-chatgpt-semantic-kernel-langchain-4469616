from langchain.chat_models import ChatOpenAI
from langchain.prompt.chat import (
  SystemMessagePromptTemplate
)
from dotenv import load_dotenv
load_dotenv()


chat_model = ChatOpenAI()


system_template = """You are a helpful assistant who generates comma
separated lists. A user will pass in a category, and you should
generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
