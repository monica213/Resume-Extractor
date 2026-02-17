from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from typing import List


# 1️⃣ Define Schema
class ResumeSchema(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    skills: List[str] = Field(description="List of skills")
    experience_years: int = Field(description="Total years of experience")
    education: List[str] = Field(description="Educational qualifications")


# 2️⃣ Initialize Ollama Model (UPDATED IMPORT)
llm = ChatOllama(
    model="llama3",
    temperature=0
)

# 3️⃣ Create JSON Parser
parser = JsonOutputParser(pydantic_object=ResumeSchema)

# 4️⃣ Create Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert resume information extraction system."),
    ("human",
     """
Extract structured information from the following resume text.

Resume:
{resume_text}

{format_instructions}
"""
     )
])


# 5️⃣ Sample Resume Text
resume_text = """
K N Monica
Email: mon@gmail.com

Skills: Python, Machine Learning, SQL, Django

Experience:
Software Engineer at ABC Company (2019-2023)
4 years experience in backend development.

Education:
B.E in Computer Science - XYZ University
"""


# 6️⃣ Format Prompt
formatted_prompt = prompt.format_messages(
    resume_text=resume_text,
    format_instructions=parser.get_format_instructions()
)

# 7️⃣ Run Model + Parse Output
try:
    response = llm.invoke(formatted_prompt)
    result = parser.parse(response.content)

    print("\n✅ Extracted Data:\n")
    print(result)

except OutputParserException as e:
    print("❌ JSON Parsing Error:", e)

except Exception as e:
    print("❌ Unexpected Error:", e)
