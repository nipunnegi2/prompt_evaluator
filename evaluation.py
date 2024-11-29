from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class EvalResponse(BaseModel):
    is_correct: bool
    reasoning: str

class Evaluator:
    def __init__(self, api_key) -> None:
        '''This is to evaluate changes to the scam_classifier_prompt against the dataset'''
        self.api_key = api_key
        self.model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.api_key)
        self.get_prompt()
        self.get_chain()

    def get_prompt(self):
        '''Define evaluation prompt statically'''
        EVAL_PROMPT_TEMPLATE = '''
        You are comparing a human ground truth answer from an expert to an answer from an AI model.
        Your goal is to determine if the AI answer correctly matches, in substance, the ground truth answer.
        BEGIN DATA]
        ***********
        Ground Truth Answer]: {human_answer}
        ***********
        AI Answer]: {ai_generated_answer}
        ***********
        END DATA]
        Compare the AI answer to the ground truth answer. If the AI correctly answers the question, then the AI answer is "correct". 
        If the AI answer is longer but contains the main idea of the ground truth answer, consider the answer "correct". 
        If the AI answer diverges and does not contain the main idea of the ground truth answer, consider the answer "incorrect".

        Return a JSON response with the keys 'is_correct' and 'reasoning'. 
        - is_correct should be True if the AI answer is correct and False if the AI answer is incorrect.
        - reasoning should include your reason as to why you chose the above.
        '''
        self.eval_prompt = ChatPromptTemplate.from_template(EVAL_PROMPT_TEMPLATE)

    def get_chain(self):
        '''Define the langchain chain to be used for evaluation'''
        self.eval_chain = self.eval_prompt | self.model.with_structured_output(EvalResponse, method="json_schema")
