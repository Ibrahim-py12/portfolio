import streamlit as st
import re
import requests

# Replace this with your actual webhook URL from IFTTT or your preferred service.
WEBHOOK_URL = 'https://connect.pabbly.com/workflow/sendwebhookdata/IjU3NjYwNTZlMDYzZjA0MzM1MjZhNTUzZDUxMzYi_pc'


def validate_email(email):
    """Basic email validation using regex."""
    pattern = r'^\S+@\S+\.\S+$'
    return re.match(pattern, email)


@st.dialog("Contact Me")
def show_contact_form():
    name = st.text_input("Name")
    email = st.text_input("Email")
    msg = st.text_area("Message")
    submit = st.button("Submit")

    if submit:
        # Validate that all fields are filled
        if not name or not email or not msg:
            st.error("Please fill in all fields.")
        # Validate the email format
        elif not validate_email(email):
            st.error("Please provide a valid email address.")
        else:
            # Prepare the data payload for the webhook
            payload = {
                'value1': name,
                'value2': email,
                'value3': msg
            }
            try:
                response = requests.post(WEBHOOK_URL, json=payload)
                if response.status_code == 200:
                    st.success("Message successfully sent!")
                else:
                    st.error("Error sending message. Please try again later.")
            except Exception as e:
                st.error(f"An error occurred: {e}")


# About Me Section
st.set_page_config(page_title="My Portfolio", page_icon="🔥", layout="wide")

col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    st.image("my_pic.jpeg", use_container_width=True)
with col2:
    st.title("M. Ibrahim Rehan")
    st.write(
        """
        AI and robotics enthusiast with expertise in Python, computer vision, and automation.  
        Passionate about building innovative solutions and real-world applications, aiming to contribute to cutting-edge technology.
        """
    )
    if st.button("📩 Contact Me"):
        show_contact_form()

st.markdown("---")

# Experience & Projects Section
st.header("Experience & Projects")

# Projects Section
st.subheader("Projects")
st.write(
    """
- **Face Detection System:** Developed a robust system using advanced computer vision techniques for real-time face detection and recognition.
- **Intelligent Automation Tool:** Built an automation tool leveraging Python to streamline repetitive tasks across various industries.
- **Robotics Integration:** Engineered AI-driven solutions to enhance robotic operations, integrating hardware with intelligent decision-making processes.
    """
)



#        streamlit run aboutme.py