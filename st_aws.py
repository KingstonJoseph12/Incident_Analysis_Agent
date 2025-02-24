import streamlit as st
from langchain_aws import ChatBedrock
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import sqlite3
import markdown
import tempfile
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.prompts import PromptTemplate
import json
import re

# Load environment variables
load_dotenv()

# Function to read and return the content of a markdown file
def read_markdown_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
# Configure Database
conn = sqlite3.connect('data.db')
cursor = conn.cursor()
db = SQLDatabase.from_uri("sqlite:///data.db")

# Initialize the ChatBedrock model
aws_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")

# Configure SQL Agent
lc_agent_executor = create_sql_agent(
    llm=aws_llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=aws_llm),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    return_intermediate_steps=True
)

# Define Monitoring Alert function
def Monitoring_Alert_func(context_input):
    """
    The LangChain Custom AI Agent is equipped with a subagent that is programmed to respond to
    Monitoring Alert incidents according to predefined instructions. This AI subagent can effectively
    and efficiently handle the identified incident type with accuracy and precision.
    """
    prompt_template = f'''
    Create a very short monitoring alert email that assures the users that it doesn't need team intervention
    and will be automatically resolved. Use key component names from the given content.
    Begin the email with
    "Hi Users,"
    End the email with
    "Regards,
    Master Data Management Team".
    Content:
    {context_input}
    '''
    
    response = aws_llm.invoke(prompt_template)
    return response.content

# Define tools for the agent
from langchain.agents import tool

@tool
def log_extraction_tool(Log_Path):
    """Using the Incident Description Extract the Logs by using the Log Directory as function parameters"""
    with open(Log_Path, 'r') as f:
        log_data = f.read()
    final_log_data = "Logs:" + log_data
    return final_log_data

@tool
def Kerberos_Issue_func(log):
    """By using the extracted Logs execute the actions as mentioned in the Knowledge Base Document
    
    ### Action 1: Force Start the Job
    Initiate a ServiceNow request via myTechExpress (mte) to force start the job. Use the following JSON template:

    ```json
    {
        "Service Catalog": "Autosys Temporary Services",
        "AA Code": "6abc",
        "Job Type": "Force Start a Box Job",
        "Job Name": "$Job_Name",
        "Description": "Force starting due to Kerberos Issue"
    }
    ```

    ### Action 2: Notify Users
    Inform users that the job has been force started and ask them to monitor the job.
    """

# Define the tools list
tools = [log_extraction_tool, Kerberos_Issue_func]

# Configure the agent with tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a powerful assistant that can extract logs, analyze them, and take appropriate actions"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Bind tools to the LLM
llm_with_tools = aws_llm.bind_tools(tools)

# Create the agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Reading markdown files
markdown_content_1 = read_markdown_file('Input_Knowledge_base.md')
markdown_content_2 = read_markdown_file('EBC_Knowledge_base.md')
markdown_content_3 = read_markdown_file('Kerberos_Knowledge_base.md')

# Extracting Information for EBC Incident
def EBC_Incident(context_input):
    """
    The LangChain custom AI agent is designed to extract important data for incident analysis from EBC_Incident descriptions.
    It efficiently identifies relevant information such as Policy ID, Expected_EMAIL_ID, Expected_Phone_Number,
    Expected_contact_name, actual_email_id, actual_phone_number, and actual_contact_name.
    """
    prompt_template = f'''
    Sample Policy Number Format: ##ABCDE####
    From the given context, extract the following information and provide them in JSON format:
    You are an automated agent. Your task is to extract the following contact details and provide them in JSON.
    Exclude if they are not present. There would not be multiple values for a single element or nested elements.
    Do not put list values inside single JSON element. Eg: Create multiple JSON files instead
    Output in JSON format, valid fields are Policy ID, expected_email_id, expected_phone_number, expected_contact_name,
    actual_email_id, actual_phone_number, actual_contact_name. Do not use markdown
    If the context doesn't have corresponding details or in that format. Just print "not applicable"
    {{
        "EBC_Incident": [
            {{
                "Policy_ID": ##ABCDE####,
                "Expected_Email_ID": "",
                "Expected_Phone_Number":"",
                "Expected_Contact_Name": "",
                "Actual_Email_ID": "",
                "Actual_Phone_Number": "",
                "Actual_Contact_Name": ""
            }},
            {{
                "Policy_ID": ##ABCDE####,
                "Expected_Email_ID": "",
                "Expected_Phone_Number":"",
                "Expected_Contact_Name": "",
                "Actual_Email_ID": "",
                "Actual_Phone_Number": "",
                "Actual_Contact_Name": ""
            }}
        ]
    }}
    Context:
    {context_input}
    '''
    
    response = aws_llm.invoke(prompt_template)
    return response.content

# Final Report Summary Generation
def report_summary_generation(EBC_Knowledge_base, result):
    """
    Generate a comprehensive incident report summary based on the analysis results and knowledge base.
    """
    prompt_template = f'''
    **Incident Report Generation for EBC Team**

    **Objective:** Summarize the findings and provide recommendations based on AI analysis of the incident (In markdown format. Do not put **Prepared by:** or **Date**).

    **Required Elements:**
    1. **Summary of the Incident:**
    - Provide a brief overview of the issue reported by the EBC Team.

    2. **Expected Cause of the Error:**
    - For each and individual policy, Outline the potential reasons behind the reported error based on AI analysis.

    3. **Recommended Next Steps:**
    - Choose one of the following actions:
        - "Good to close the Incident"
        - "Further analysis from MDM team is needed"

    **Special Instructions:**
    - If there is a discrepancy between the expected contact and the actual contact, and the database shows the same discrepancy, explain why the MDM Database reflects the actual contact.
    - If the actual contact differs from the expected contact but the database analysis suggests they should match, recommend further investigation by the MDM team to determine the cause of the inconsistency.

    Use the reference document to assist in creating summary
    **Reference Document:**
    - **Knowledge Base**:
    {EBC_Knowledge_base}

    **Analysis Output:**
    - {result}
    '''
    
    response = aws_llm.invoke(prompt_template)
    return response.content


def log_analysis_summary(response_summary, htmlmarkdown):
    """
    Generate a JSON-formatted summary of actions based on the analysis and knowledge base.
    """
    prompt_template = f'''
    Using the following Knowledge base. Provide the actions for the below Analysis in JSON format. Only include JSON output
    <KnowledgeBase>
    {htmlmarkdown}
    </KnowledgeBase>

    <AnalysisSummary>
    - {response_summary}
    </AnalysisSummary>
    '''
    
    response = aws_llm.invoke(prompt_template)
    return response.content
# Creating the sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Incident Analyzer", "Input Knowledge Base", "EBC Knowledge Base","Kerberos Knowledge Base"])

# Display content based on the selected page
if page == "Input Knowledge Base":
    st.title("Input Knowledge Base")
    st.markdown(markdown_content_1)

elif page == "EBC Knowledge Base":
    st.title("EBC Knowledge Base")
    st.markdown(markdown_content_2)
    
elif page == "Kerberos Knowledge Base":
    st.title("Kerberos Knowledge Base")
    st.markdown(markdown_content_3)

elif page == "Incident Analyzer":
    st.title("Incident Analyzer")
    st.write("Welcome to Automated Incident Analyzer! Our advanced tool effortlessly integrates with your data sources and Incident Management systems to deliver specialized data analysis tailored to your domain. Not only does it generate comprehensive reports, but it also autonomously resolves incidents. We've prepared two sample scenarios with detailed descriptions for demonstration purposes. In real-world applications, incident descriptions are automatically pulled from the Incident Management Tool whenever a new incident is triggered. This highly scalable system can be customized to manage numerous incidents across various domains and teams. For further information on the sample incidents, please visit our Knowledge Base via the left panel navigation.")
    st.subheader("Choose a Sample Incident")
    # Input area for incident description
    default_text = st.text_area("Incident Description")
    user_input = default_text
    
    if st.button("Submit"):

        # Load the knowledge bases
        with open('Input_Knowledge_base.md', 'r', encoding='utf-8') as f:
            Input_Knowledge_base = markdown.markdown(f.read())

        with open('EBC_Knowledge_base.md', 'r', encoding='utf-8') as file:
            EBC_Knowledge_base = markdown.markdown(file.read())

        Monitoring_Alert = '''
            "Hi Users,"

            [AUTOMATIC RESOLUTION NOTICE: The alert has been received and is expected to be resolved automatically. No team intervention is required at this time.]

            Regards,
            Master Data Management Team
        '''

        # Simulating incident classification
        
        def classify_incident(incident):
            """
            Classify the incident type based on the knowledge base and incident description.
            """
            with open('Input_Knowledge_base.md', 'r', encoding='utf-8') as f:
                htmlmarkdown = markdown.markdown(f.read())
            
            prompt_template = f'''
            Provide the incident context and a knowledge base document in HTML markdown format.
            The incident type value will be in the headers of the knowledge base.
            The context will only have one incident type value.
            
            Incident:
            {incident}

            Knowledge base document
            {htmlmarkdown}

            Output should only have the incident type in format as follows (Do not include markdowns ```json):
            {{"Incidents": {{
                "Incident_Type": INCIDENT_TYPE_VALUE
                }}
            }}

            If the context does not match any incident type, output "Not_applicable".
            '''
            
            response = aws_llm.invoke(prompt_template)
            output_string = re.sub(r'^\`\`\`json\n|\n\`\`\`$', '', response.content, flags=re.MULTILINE)
            
            st.subheader("Classifier Agent")
            st.json(output_string)
            
            data = json.loads(output_string)
            return data['Incidents']['Incident_Type']
        Incident = user_input

        # Process the incident based on its classification
        incident_type = classify_incident(Incident)

        if incident_type == "EBC_Incident":
            # Handle EBC incident
            first_response = EBC_Incident(Incident)
            prompt = f"For the below data do the analysis for mismatch \nData: \n{first_response}\n using the Knowledge Base:{EBC_Knowledge_base}"
            
            st.write("The Custom AI Agent Analysis (Claude 3.5 - Sonnet)")
            with st.container(border=True):
                st_callback = StreamlitCallbackHandler(st.container())
                response = lc_agent_executor.invoke(
                    {"input": prompt}, {"callbacks": [st_callback]}
                )
            
            st.subheader("Response Summary:")
            with st.spinner('Summarizing the Analysis Results'):
                final_summary = report_summary_generation(EBC_Knowledge_base, response["output"])
                final_summary = re.sub(r'^\`\`\`markdown\n|\n\`\`\`$', '', final_summary, flags=re.MULTILINE)

                st.download_button("Download Response Summary", final_summary)
                with st.expander(label="Summary", expanded=True):
                    st.markdown(final_summary)
        elif incident_type == "Monitoring_Alert":
            # Handle Monitoring Alert incident
            with st.spinner("Resolving the monitoring alert"):
                result = Monitoring_Alert_func(Incident)
                st.subheader("Generated Response:")
                st.write(result)
                st.subheader("Next Steps:")
                st.write("Close the Incident after notifying the corresponding users")
        elif incident_type == "Autosys_Issue":
            # Handle Autosys Issue incident
            input_kb = read_markdown_file('Kerberos_Knowledge_base.md')

            prompt = f'''
            For the below data do the analysis \nData: myTechExpress ASSGNED:SEV4 Incident INC000011119999 6abcp5cmd_ag_log OPSREPORTING\n\n using the Knowledge Base:{input_kb}
            '''
            
            st.write("The Custom Autosys Bot Analysis (Claude 3.5 Sonnet)")
            with st.container(border=True):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent_executor.invoke(
                    {"input": prompt}, {"callbacks": [st_callback]}
                )
                
            with st.spinner("Analyzing the next steps"):
                with open('Kerberos_Knowledge_base.md', 'r', encoding='utf-8') as f:
                    htmlmarkdown = markdown.markdown(f.read())
                
                final_summary = log_analysis_summary(response["output"], htmlmarkdown)
                st.subheader("Next Steps:")
                output_string = re.sub(r'^\`\`\`json\n|\n\`\`\`$', '', final_summary, flags=re.MULTILINE)
                st.json(output_string)