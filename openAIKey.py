import os
from dotenv import load_dotenv


from openai import OpenAI
cus_path = os.getcwd() + "/venv/bin/.env"
load_dotenv(cus_path)
print(os.getenv("OPENAI_API_KEY"))