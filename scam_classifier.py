from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class ScamResponse(BaseModel):
    confidence_score: float
    scam_category: str

class ClassifyScam:
    def __init__(self, api_key) -> None:
        '''Classify messages as scams with confidence and type'''
        self.api_key = api_key
        self.model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.api_key)
        self.get_prompt()
        self.get_chain()

    def get_prompt(self):
        '''Define scam classifier prompt statically'''
        SCAM_PROMPT_TEMPLATE = '''
        You are an expert forensics analyst specializing in image forensics. Your current task is to match the user-provided image and determine whether it is a scam or not. 
        Always include a confidence score between 0 to 1 to suggest how confident you are about your answer.

        In case you are confident that it is a scam (score of over 0.7), classify the scam into one of the following categories:

        1. impersonation_scam
        2. cyber_bullying
        3. email_phishing
        4. fake_parcel_scam
        5. online_job_scam
        6. matrimonial_scam
        7. credit_card_fraud
        8. demat_stocks_scam
        9. ewallet_fraud
        10. fraud_voice_calls
        11. internet_banking_scam
        12. upi_fraud

        Return the response as a JSON object.
        '''
        self.prompt = ChatPromptTemplate.from_template(SCAM_PROMPT_TEMPLATE)

    def get_chain(self):
        '''Define the langchain chain to be used for classification'''
        self.scam_classifier_chain = self.prompt | self.model.with_structured_output(ScamResponse, method="json_schema")

    def invoke(self, input_text):
        '''Invoke the chain with input text'''
        return self.scam_classifier_chain.invoke({'input_text': input_text})
