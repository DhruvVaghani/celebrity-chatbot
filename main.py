# ## Integrating code with OpenAI API
# import os
# import streamlit as st
# openai_key = st.secrets["OPENAI_API_KEY"]
# #from langchain_community.llms import ChatOpenAI 
# import streamlit as st
# #from langchain import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains import SequentialChain
# from langchain.memory import ConversationBufferMemory
# import wikipedia
# #from langchain.chat_models import ChatOpenAI
# import urllib.parse
# from langchain_core.prompts import PromptTemplate
# from langchain_community.chat_models import ChatOpenAI

import os
import streamlit as st
import wikipedia


from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory


openai_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_key

# Step 1: Autocomplete using Wikipedia API
typed_name = st.text_input("Search for a celebrity")

if typed_name and len(typed_name) > 2:
    try:
        suggestions = wikipedia.search(typed_name)
        if suggestions:
            selected_name = st.selectbox("Did you mean?", suggestions)
            st.session_state.selected_name = selected_name 
        else:
            st.warning("No suggestions found.")
            st.session_state.selected_name = typed_name
    except Exception as e:
        st.error(f"Error fetching suggestions: {e}")
        st.session_state.selected_name = typed_name
else:
    selected_name = None


 


# Step 2: Proceed when user clicks search
if st.button("Search") and "selected_name" in st.session_state:
    selected_name = st.session_state.selected_name
    st.success(f"Fetching details for **{selected_name}**")


    # 🎥 YouTube Search Link (instead of embedding video)
    search_query = f"{selected_name} interview"
    youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"

    st.subheader("🎥 YouTube Interviews & Clips")
    st.markdown(f"[Click here to watch on YouTube]({youtube_url})", unsafe_allow_html=True)


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
    llm =ChatOpenAI(temperature = 0.8,model_name="gpt-4", max_tokens=300)
    chain = LLMChain(llm=llm, prompt = first_input_prompt, verbose = True, output_key = 'person', memory = person_memory)






    #Prompt Templates
    second_input_prompt = PromptTemplate( 
        input_variable = ['person'], 
        template = 'When was {person} born'
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


    # Run the chain
    output = parent_chain({'name': selected_name})

        # Show outputs
    st.subheader("🔍 Person Details")
    st.write(output['person'])

    st.subheader("🎂 Date of Birth")
    st.write(output['dob'])

    st.subheader("🌍 Major Events Around Birth")
    st.write(output['majorevents'])

        # Show memory buffers
    with st.expander("🧠 Person Memory"):
        st.info(person_memory.buffer)
    with st.expander("📅 Events Memory"):
        st.info(majorevents_memory.buffer)