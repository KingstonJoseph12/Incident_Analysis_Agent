import streamlit as st
from langchain_aws import ChatBedrock
from langchain_community.utilities import SQLDatabase
# from langchain_openai import ChatOpenAI
# from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
# from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
# from langchain_openai import AzureOpenAI
# from streamlit_extras.switch_page_button import switch_page
import os
# from helpers.llm_helper import chat, stream_parser
from openai import AzureOpenAI
from dotenv import load_dotenv
import sqlite3
import markdown
# import pdfkit
import tempfile
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import json
# import time
import re

#Configuring Azure OpenAI
load_dotenv()


# ''' Azure Connection
# client = AzureOpenAI(
#     api_key=os.environ.get('GPT_4o_KEY'),  
#     api_version="2024-02-15-preview",
#     azure_endpoint = 'https://openai-hartford-cloudboost.openai.azure.com/'
#     )
    
# deployment_name='hartford-gpt-4o'

# os.environ["OPENAI_API_VERSION"]= "2024-02-15-preview"
# os.environ["AZURE_OPENAI_ENDPOINT"] =  "https://openai-hartford-cloudboost.openai.azure.com/"
# os.environ["AZURE_OPENAI_API_KEY"] = os.environ.get('GPT_4o_KEY')
# '''

# Function to read and return the content of a markdown file
def read_markdown_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
#Configure Database
 
# Connect to the database
conn = sqlite3.connect('data.db')
# Create a cursor
cursor = conn.cursor()
db = SQLDatabase.from_uri("sqlite:///data.db")

# ''' Azure Langchain Connection
# llm = AzureChatOpenAI(deployment_name='hartford-gpt-4o',verbose=True,temperature=0)
# '''

# Initialize the ChatBedrock model
aws_llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")

#Configure Agent
lc_agent_executor = create_sql_agent(
    llm=aws_llm,
    toolkit=SQLDatabaseToolkit(db=db,llm=aws_llm),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # agent_type='openai-tools',
    handle_parsing_errors = True,
    verbose=True,
    return_intermediate_steps=True,
    # callback_manager=
)

#Define Monitoring Alert

def Monitoring_Alert_func(context_input):
    '''
     The LangChain Custom AI Agent is equipped with a subagent that is programmed to respond to Monitoring Alert incidents according to predefined instructions. This AI subagent can effectively and efficiently handle the identified incident type with accuracy and precision.
    '''
    prompt_array = []
    prompt_template = f'''
    Create a very short monitoring alert email that assures the users that it doesn't need team intervention and will be automatically resolved.  Use key component names from the given content.
    Begin the email with
    "Hi Users,"
    End the email with
    "Regards,
    Master Data Management Team".
    Content:
    {context_input}    
    '''
    complete_prompt = {"role": "user", "content": prompt_template}
    prompt_array.append(complete_prompt)
    
    response = aws_llm.invoke(prompt_template)

    # completion = client.chat.completions.create(model=deployment_name, messages=prompt_array,max_tokens=1000)
    # response = completion.choices[0].message.content
    # print("Generated Mail:" ,completion.choices[0].message.content)
    return response.content

from langchain.agents import tool


@tool
def log_extraction_tool(Log_Path):
    '''Using the Incident Description Extract the Logs by using the Log Directory as function parameters'''
    with open(Log_Path, 'r') as f:
        log_data = f.read()
    final_log_data = "Logs:"+log_data
    # st.subheader("Latest Logs:")
    # st.write(final_log_data)
    # final_information = [output_string,final_log_data]
    # print("Generated Mail:" ,completion.choices[0].message.content)
    return final_log_data

@tool
def Kerberos_Issue_func(log):
    '''By using the extracted Logs execute the actions as mentioned in the Knowledge Base Document
    
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
'''

tools=[log_extraction_tool,Kerberos_Issue_func]
# def Kerberos_Issue_func(incident,log):
#     prompt_array = []
#     markdown_content_ker = read_markdown_file('Kerberos_Knowledge_base.md')
#     prompt_template = f'''
#     {markdown_content_ker}
#     Incident:
#     {incident}
#     Logs:
#     {log}
#     '''
#     complete_prompt = {"role": "user", "content": prompt_template}
#     prompt_array.append(complete_prompt)
       
#     completion = client.chat.completions.create(model=deployment_name, messages=prompt_array,max_tokens=1000)
#     response = completion.choices[0].message.content
#     # print("Generated Mail:" ,completion.choices[0].message.content)
#     return response   
    
# def log_extraction_tool(inc_desc):
#     input_kb = read_markdown_file('Kerberos_Knowledge_base.md')
#     prompt_array=[]
#     prompt_template= f'''
#     For the autosys failure, extract the log path from the knowledge base and provide it in JSON format
#     Output should only have the Job name and Log Path type in format as follows (Do not include markdowns ```json):
#             {{"Autosys_Failure": {{
#                 "Incident_Description": Incident Description Mentioned Below
#                 "Incident_Number": INC##### Format in Incident Description
#                 "Job_Name": Extract Job Name from the Incident
#                 "Log_Path":LOG PATH VALUE
#                 }}
#                 }}
#     Knowledge_Base:
#     {input_kb}
#     Incident_Description:
#     {inc_desc}
#     '''
#     complete_prompt = {"role": "user", "content": prompt_template}
#     prompt_array.append(complete_prompt)
       
#     completion = client.chat.completions.create(model=deployment_name, messages=prompt_array,max_tokens=1000)
#     response = completion.choices[0].message.content
    
#     output_string = re.sub(r'^\`\`\`json\n|\n\`\`\`$', '', response, flags=re.MULTILINE)
#     st.subheader("Extracted_Data")
#     st.json(output_string)
#     data=json.loads(output_string)
    
#     # Your logic for classifying the incident goes here
#     # This could be based on keyword matching, pattern recognition, or any other suitable method.
#     Log_Path = data['Autosys_Failure']['Log_Path']
    
#     with open(Log_Path, 'r') as f:
#         log_data = f.read()
#     final_log_data = "Logs"+log_data
#     st.subheader("Latest Logs:")
#     st.write(final_log_data)
#     # final_information = [output_string,final_log_data]
#     # print("Generated Mail:" ,completion.choices[0].message.content)
#     return output_string, final_log_data       
    

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, You can extract the logs and analyze the logs and take actions",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = aws_llm.bind_tools(tools)

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Reading markdown files
markdown_content_1 = read_markdown_file('Input_Knowledge_base.md')
markdown_content_2 = read_markdown_file('EBC_Knowledge_base.md')
markdown_content_3 = read_markdown_file('Kerberos_Knowledge_base.md')

#Extracting Information for EBC Incident

def EBC_Incident(context_input):
    '''
    The LangChain custom AI agent is designed to extract important data for incident analysis from EBC_Incident descriptions. It efficiently identifies relevant information such as Policy ID, Expected_EMAIL_ID, Expected_Phone_Number, Expected_contact_name, actual_email_id, actual_phone_number, and actual_contact_name.
    '''
    prompt_array = []
    prompt_template = f'''
    Sample Policy Number Format: ##ABCDE####
    From the given context, extract the following information and provide them in JSON format:
    You are an automated agent. Your task is to extract the following contact details and provide them in JSON.
    Exclude if they are not present. There would not be multiple values for a single element or nested elements. Do not put list values inside single JSON element. Eg:  Create multiple JSON files instead
    Output in JSON format, valid fields are Policy ID, expected_email_id, expected_phone_number, expected_contact_name, actual_email_id, actual_phone_number, actual_contact_name. Do not use markdown
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
    complete_prompt = {"role": "user", "content": prompt_template}
    prompt_array.append(complete_prompt)
    
    response = aws_llm.invoke(prompt_template)

    first_response = response.content
    # completion = client.chat.completions.create(model=deployment_name, messages=prompt_array)
    # first_response = completion.choices[0].message.content
    return first_response

#Final Report Summary Generation

def report_summary_generation(EBC_Knowledge_base,result):
    prompt_array = []
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
    complete_prompt = {"role": "user", "content": prompt_template}
    prompt_array.append(complete_prompt)
    
    response = aws_llm.invoke(prompt_template)

    final_analysis_summary = response.content
    # completion = client.chat.completions.create(model=deployment_name, messages=prompt_array)
    # final_analysis_summary = completion.choices[0].message.content
    return final_analysis_summary


def log_analysis_summary(response_summary,htmlmarkdown):
    prompt_array = []
    prompt_template = f'''
    Using the following Knowledge base. Provide the actions for the below Analysis in JSON format. Only include JSON output
    <KnowledgeBase> 
    {htmlmarkdown}
    </KnowledgeBase>

    <AnalysisSummary>
    - {response_summary}
    </AnalysisSummary>
    '''
    complete_prompt = {"role": "user", "content": prompt_template}
    prompt_array.append(complete_prompt)
    
    response = aws_llm.invoke(prompt_template)

    final_analysis_summary = response.content
    # completion = client.chat.completions.create(model=deployment_name, messages=prompt_array)
    # final_analysis_summary = completion.choices[0].message.content
    return final_analysis_summary

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
    # Custom texts for each radio button option
    default_text = st.text_area("Incident Description" )
    # tab1, tab2 = st.tabs(["📈 Monitoring Alert", "📨 Contact Mismatch"])
    # Display radio buttons and store the selected option
    # selected_option = st.radio("Choose an option:", ("Monitoring Incident", "EBC Contact Incident","Kerberos Issue"))
    # if selected_option == "Monitoring Incident":
    #     default_text = text_option_1
    # elif selected_option == "EBC Contact Incident":
    #     default_text = text_option_2
    # elif selected_option == "Kerberos Issue":
    #     default_text = text_option_3
    # st.code(default_text,language="markdown")
    user_input= default_text
    # user_input = st.text_area("Enter incident description:", placeholder="The incident description will be automatically retrieved from the Incident Management Tool in a real scenario.",value=default_text)
    
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
            f = open('Input_Knowledge_base.md', 'r', encoding='utf-8')
            htmlmarkdown=markdown.markdown(f.read())
        
            Message_Array=[]
        
            context_input = incident
        
            prompt_template = f'''
            Provide the incident context and a knowledge base document in HTML markdown format. The incident type value will be in the headers of the knowledge base. The context will only have one incident type value.
            Incident:
            {context_input}

            Knowledge base document
            {htmlmarkdown}

            Output should only have the incident type in format as follows (Do not include markdowns ```json):
            {{"Incidents": {{
                "Incident_Type":INCIDENT_TYPE_VALUE
                }}
                }}

            If the context does not match any incident type, output "Not_applicable".
            '''
            complete_prompt = {"role": "user", "content": prompt_template}
            Message_Array.append(complete_prompt)
            
            response = aws_llm.invoke(prompt_template)
            response = response.content
            # completion = client.chat.completions.create(model='hartford-gpt-4o', messages=Message_Array)
            
            # response = completion.choices[0].message.content
            # time.sleep(5)
            output_string = re.sub(r'^\`\`\`json\n|\n\`\`\`$', '', response, flags=re.MULTILINE)
            st.subheader("Classifier_Agent")
            # st.write(output_string)
            st.json(output_string)
            # print(output_string)
            data=json.loads(output_string)
            
            # Your logic for classifying the incident goes here
            # This could be based on keyword matching, pattern recognition, or any other suitable method.
            Incident_Type = data['Incidents']['Incident_Type']
            return(Incident_Type)
        Incident = user_input

        # Incident = input('''
        #                 Enter the description of the incident:
        #                 ''')
        
        incident_type = classify_incident(Incident)

        # if incident_type == "EBC_Incident":
        #     prompt = f"{Input_Knowledge_base}\n{EBC_Knowledge_base}\nIncident:\n{Incident}\n"
        if incident_type == "EBC_Incident":
            first_response = EBC_Incident(Incident)
            prompt = f"For the below data do the analysis for mismatch \nData: \n{first_response}\n using the Knowledge Base:{EBC_Knowledge_base}"
            # result = lc_agent_executor.invoke(prompt)
            # with st.expander("Agent Thoughts"):
            st.write("The Custom AI Agent Analysis (Claude 3.5 - Sonnet)")
            with st.container(border=True):
                st_callback = StreamlitCallbackHandler(st.container())
                response = lc_agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
            )
            # st.expander()
            st.subheader("Response Summary:")
            with st.spinner('Summarizing the Analysis Results'):
                final_summary = report_summary_generation(EBC_Knowledge_base,response["output"])
                final_summary = re.sub(r'^\`\`\`markdown\n|\n\`\`\`$', '', final_summary, flags=re.MULTILINE)

                st.download_button("Download Response Summary", final_summary)
                with st.expander(label="Summary", expanded=True):
                    st.markdown(final_summary)
        elif incident_type == "Monitoring_Alert":
            
            # prompt = f"{Input_Knowledge_base}\n{Monitoring_Alert}\nIncident:\n{Incident}\n"
            # result = agent_executor.invoke(prompt)
            with st.spinner("Resolving the monitoring alert"):
                result = Monitoring_Alert_func(Incident)
                st.subheader("Generated Response:")
                st.write(result)
                st.subheader("Next Steps:")
                st.write("Close the Incident after notifying the corresponding users")
        elif incident_type == "Autosys_Issue":
            input_kb = read_markdown_file('Kerberos_Knowledge_base.md')

            prompt = f'''
            For the below data do the analysis  \nData: myTechExpress ASSGNED:SEV4 Incident INC000011119999 6abcp5cmd_ag_log OPSREPORTING\n\n using the Knowledge Base:{input_kb}
            '''
            
            st.write("The Custom Autosys Bot Analysis (Claude 3.5 Sonnet)")
            with st.container(border=True):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
            )
                
            with st.spinner("Analyzing the next steps"):
                f = open('Kerberos_Knowledge_base.md', 'r', encoding='utf-8')
                htmlmarkdown=markdown.markdown(f.read())
                summary = htmlmarkdown
                final_summary = log_analysis_summary(response["output"],htmlmarkdown)
                st.subheader("Next Steps:")
                output_string = re.sub(r'^\`\`\`json\n|\n\`\`\`$', '', final_summary, flags=re.MULTILINE)
                st.json(output_string)
                
            # with st.spinner("Data_Extraction in progress"):
            #     incident_desc,log_data = log_extraction_tool(Incident)
            # with st.spinner("Analyzing the Logs for next course of action"):
            #     final_response = Kerberos_Issue_func(incident_desc,log_data)
            #     st.write(final_response)