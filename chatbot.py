import google.generativeai as genai
import pandas as pd
import json
import typing_extensions
import streamlit as st

# Configure Gemini
genai.configure(api_key="AIzaSyCNVJFXxKWhJbbx2dyNzz9ns1weyk4SusU")
model_pandas = genai.GenerativeModel('gemini-1.5-flash', system_instruction="You are an expert python developer who works with pandas. You make sure to generate simple pandas 'command' for the user queries in JSON format. No need to add 'print' function. Analyse the datatypes of the columns before generating the command. If unfeasible, return 'None'. ")
model_response = genai.GenerativeModel('gemini-1.5-flash', system_instruction="Your task is to comprehend. You must analyse the user query and response data to generate a response data in natural language.")
class Command(typing_extensions.TypedDict):
    command: str

def app():
    st.title('Chatbot')
    st.write('Bicaralah dengan data Anda')

    # Check if DataFrame is in session_state
    if 'dataframe' not in st.session_state:
        st.error("No data loaded. Please upload data via the 'Google Sheet Link' page first.")
        return

    df = st.session_state['dataframe']
    head = str(df.head().to_dict())
    desc = str(df.describe().to_dict())
    cols = str(df.columns.to_list())
    dtype = str(df.dtypes.to_dict())

    # User Query
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Apa yang bisa saya bantu?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        final_query = f"The dataframe name is 'df'. df has the columns {cols} and their datatypes are {dtype}. df is in the following format: {desc}. The head of df is: {head}. You cannot use df.info() or any command that cannot be printed. Write a pandas command for this query on the dataframe df: {user_query}"

        with st.spinner('Analyzing the data...'):
            response = model_pandas.generate_content(
                final_query,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=Command,
                    temperature=0.3
                )
            )
            command = json.loads(response.text)['command']
            print(command)
        try:
            # Consider using eval or a limited interpreter for pandas commands
            result = eval(command, {"df": df})  # Assuming df is safe

            
            if result is None:
                natural_response = "Silakan ajukan pertanyaan untuk memulai."
            else:
                # Display the result as a Streamlit table
                st.write(result)  # Assuming the result is a pandas DataFrame

                natural_response = f"Pertanyaan pengguna adalah '{user_query}'. Hasilnya ditampilkan di atas."

            bot_response = model_response.generate_content(
                natural_response,
                generation_config=genai.GenerationConfig(temperature=0.7)
            )

            st.chat_message("assistant").write(bot_response.text)
            st.session_state.messages.append({"role": "assistant", "content": bot_response.text})

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": "error"})

if __name__ == "__main__":
    app()