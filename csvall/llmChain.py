import warnings
import os
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI
from langchain_aws import ChatBedrock
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from crewai_tools import tool
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import load_tools
from crewai import Crew, Process, Agent, Task
import json
import re

warnings.filterwarnings('ignore')



#Semantic similarity tool:
def findBestAnswer(prompt):
    @tool("Similarity Search")
    def similarity_search(response_1: str, response_2: str, response_3: str, user_query: str) -> str:
        """This tool uses a semantic similarity search to determine which of three responses (response_1, response_2, and response_3) is the most similar to a user's query (user_query). It chooses the prompt with the highest simiarity score"""

        #Prompt template that LLM response will be formatted into
        example_prompt = PromptTemplate(
            input_variables=["response"],
            template="Response: {response}",
        )

        #Attaching responses from all 3 LLMs for similarity search
        examples = [
            {"response": response_1},
            {"response": response_2},
            {"response": response_3}
        ]

        #Selector for greatest cosine similarity
        similarity_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embeddings,
            vectorstore,
            k=1
        )

        #Prompt template that contains few shot examples (responses from llms)
        similar_prompt = FewShotPromptTemplate(
            example_selector=similarity_selector,
            example_prompt=example_prompt,
            prefix="Chose the most similar response to the user query:",
            suffix="User query: {user_query}:",
            input_variables=["user_query"],
        )

        return similar_prompt.format(user_query=user_query)

    serper = load_tools(["serpapi"])

    #Anthropic
    researcher_1 = Agent(
        role='Anthropic Researcher',
        goal="Provide answer to questions. Answer any questions you are capable of answering without any tools. ONLY use serper tool as a last resort for current/future events that you do not have the knowledge to answer. You can assume the serper tool is correct. You cannot use context, you must find your own answer.",
        backstory='A meticulous researcher that thoroughly answers questions.',
        verbose=True,
        llm=anthropic,
        allow_delegation=False,
        tools=serper,
        cache=False
    )

    #Vertex
    researcher_2 = Agent(
        role='Vertex Researcher',
        goal='Provide answer to questions. Answer any questions you are capable of answering without any tools. ONLY use serper tool as a last resort for current/future events that you do not have the knowledge to answer. You cannot use context, you must find your own answer.',
        backstory='A meticulous researcher that thoroughly answers questions.',
        verbose=True,
        llm=vertex,
        allow_delegation=False,
        tools=serper,
        cache=False
    )

    #OpenAI
    researcher_3 = Agent(
        role='OpenAI Researcher',
        goal='Provide answer to questions. Answer any questions you are capable of answering without any tools. ONLY use serper tool as a last resort for current/future events that you do not have the knowledge to answer. You cannot use context, you must find your own answer.',
        backstory='A meticulous researcher that thoroughly answers questions.',
        verbose=True,
        llm=openai,
        allow_delegation=False,
        tools=serper,
        cache=False
    )

    manager = Agent(
        role="Manager",
        goal="Using the entirety of the final results of the responses from Anthropic Researcher, Vertex Researcher, and OpenAI Researcher, use the similarity search tool to calculate which has the most similar response to the original user query. Provide the similarity score ranking in your response.",
        backstory="A meticulous manager that thoroughly analyzes research results.",
        verbose=True,
        llm=vertex,
        allow_delegation=False,
        tools=[similarity_search]
    )

    user_input = prompt
    research_task1 = Task \
        (description="Answer the following question from the user. You cannot use context provided to answer, you must find your own. Do not use any tools for fact-based questions about a past event. Use your own knowledge base as a priority." + user_input,
         agent=researcher_1, expected_output='String answer to user query')
    research_task2 = Task \
        (description="Answer the following question from the user. You cannot use context or knowledge from previous researchers to answer, you must find your own. Do not use any tools for fact-based questions about a past event. Use your own knowledge base as a priority." + user_input,
         agent=researcher_2, expected_output='String answer to user query')
    research_task3 = Task \
        (description="Answer the following question from the user. You cannot use context or knowledge from previous researchers to answer, you must find your own. Do not use any tools for fact-based questions about a past event. Use your own knowledge base as a priority." + user_input,
         agent=researcher_3, expected_output='String answer to user query')

    #Manager
    manager_task = Task \
        (description="Using the final raw outputs from Anthropic Researcher, Vertex Researcher, and OpenAI Researcher (research_task1.output.raw_output, research_task2.output.raw_output, and research_task3.output.raw_output), you MUST use the similarity search tool to calculate which has the most similar response to the original user query (" + user_input + "). Use the ENTIRE final output from each of the three researchers as string input into the similarity search tool along with the user input. Output the ENTIRETY of the most similar response ALONG with the similarity score and which researcher was selected.",
         agent=manager,
         expected_output="String answer to user query with the highest similarity score along with similarity score and which researcher was selected")

    #Creating sequential chain of agents
    crew = Crew(
        agents=[researcher_1, researcher_2, researcher_3, manager],
        tasks=[research_task1, research_task2, research_task3, manager_task],
        process=Process.sequential,
    )

    # Start the crew's work
    result = crew.kickoff()
    task1_output = research_task1.output
    final_output_Anthropic = task1_output.raw_output
    print("Output from Anthropic")
    print(final_output_Anthropic)
    task2_output = research_task2.output
    final_output_vertex = task2_output.raw_output
    print("Output from Vertex")
    print(final_output_vertex)
    task3_output = research_task3.output
    final_output_openai = task3_output.raw_output
    print("Output from Azure OpenAI")
    print(final_output_openai)
    print("---------------")
    print("Most Relevant Response")
    task4_output = manager_task.output
    final_output_similarity = task4_output.raw_output

    llm_name = final_output_similarity  #
    best_answer = final_output_similarity  #
    # Extract the JSON portion using regex
    pattern = r'json\s+({.*?})'
    match = re.search(pattern, final_output_similarity, re.DOTALL)

    if match:
        json_str = match.group(1)  # Extract the JSON string
        json_object = json.dumps(json_str)  # Convert to JSON object
        json_str += "}"
        json_object = json.loads(json_str)
        best_answer = json_object["most_similar_response"]["content"]
        llm_name = json_object["most_similar_response"]["researcher"].strip("Researcher")
    return llm_name, best_answer


# llmChain.py


if __name__ == '__main__':
    exit()
