import streamlit as st

# Replace this with the actual credentials or mechanism to fetch from Google Sheets
credentials = {
    "stevensamuell177@gmail.com": "AlongAlong",
    "kelompok43@gmail.com": "Kelompok43"
}

st.warning("Please login with username: kelompok43@gmail.com and password: Kelompok43. Authentication is still hardcoded due to cloud traffic.")

def app():
    st.title('Welcome to :blue[BizTrack]')
     
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''

    def login():
        email = st.session_state.email_input
        password = st.session_state.password_input
        if email in credentials and credentials[email] == password:
            st.session_state.username = email.split('@')[0]
            st.session_state.useremail = email
            st.session_state.signedout = True
            st.session_state.signout = True
            st.success('Login successful')
        else:
            st.warning('Login Failed')

    def logout():
        st.session_state.signout = False
        st.session_state.signedout = False
        st.session_state.username = ''
        st.session_state.useremail = ''

    def forget():
        st.info('Password reset is not supported in this demo')

    if "signedout" not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False

    if not st.session_state["signedout"]:
        choice = st.selectbox('Login/Signup', ['Login'])
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')
        st.session_state.email_input = email
        st.session_state.password_input = password

        if choice == 'Login':
            st.button('Login', on_click=login)
            

    if st.session_state.signout:
        st.text('Name: ' + st.session_state.username)
        st.text('Email id: ' + st.session_state.useremail)
        st.button('Sign out', on_click=logout)
