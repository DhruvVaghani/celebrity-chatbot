## Integrating code with OpenAI API
import os
from constants import openai_key
from langchain_community.llms import OpenAI 
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"] = openai_key

#Streamlit framework
st.title('Search for a Celebrity')
input_text = st.text_input('Search the topic')

#Memory
person_memory = ConversationBufferMemory(input_key = 'name', memory_key = 'chat_history')
dob_memory = ConversationBufferMemory(input_key = 'person', memory_key = 'chat_history')
majorevents_memory = ConversationBufferMemory(input_key = 'dob', memory_key = 'majorevents_history')

#Prompt Templates
first_input_prompt = PromptTemplate( 
    input_variable = ['name'], 
    template = 'Tell me about {name}'
)

##OPEN AI LLMS
llm =OpenAI(temperature = 0.8,)
chain = LLMChain(llm=llm, prompt = first_input_prompt, verbose = True, output_key = 'person', memory = person_memory)






#Prompt Templates
second_input_prompt = PromptTemplate( 
    input_variable = ['person'], 
    template = 'When was {person} bron'
)
##OPEN AI LLMS
chain2 = LLMChain(llm=llm, prompt = second_input_prompt, verbose = True, output_key = 'dob', memory = dob_memory)


#Prompt Templates
third_input_prompt = PromptTemplate( 
    input_variable = ['dob'], 
    template = 'Mention 2 major events that took place around {dob} in the world'
)
##OPEN AI LLMS
chain3 = LLMChain(llm=llm, prompt = third_input_prompt, verbose = True, output_key = 'majorevents', memory = majorevents_memory)

parent_chain =SequentialChain(chains=[chain,chain2,chain3], input_variables =['name'], output_variables=['person', 'dob', 'majorevents'], verbose = True)


if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(majorevents_memory.buffer)
