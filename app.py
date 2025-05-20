import streamlit as st
import os

# Must precede any llm module imports
from langtrace_python_sdk import langtrace
langtrace.init(api_key = st.secrets['LANGTRACE_API_KEY'])

from chains import rag_chain
from history import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import time
import json
from openai import OpenAI

def validate_api_key(api_key):
    """Validate the API key format and test it"""
    if not api_key:
        return False, "API key cannot be empty"

    # Check if it starts with 'sk-' and has appropriate length
    if not api_key.startswith('sk-') or len(api_key) < 20:
        return False, "Invalid API key format. OpenAI API keys start with 'sk-' and are at least 20 characters long."

    try:
        # Test the API key with a simple request
        client = OpenAI(api_key=api_key)
        client.models.list()  # This will fail if the API key is invalid
        return True, "API key is valid"
    except Exception as e:
        return False, f"Invalid API key: {str(e)}"

# Initialize API key in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]

def update_api_key(new_api_key):
    """Update the API key in session state after validation"""
    is_valid, message = validate_api_key(new_api_key)
    if is_valid:
        st.session_state.api_key = new_api_key
        st.secrets["OPENAI_API_KEY"] = new_api_key
        return True, message
    return False, message

# Original RAG chain setup
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# List of predefined Q&A pairs for evaluation
questions_answers = [
    # Easy
    ("Which courses can I take first semester?",
     "You can take CET 1100, CET 1111, CET 1120, CET 1150, ENG 1101, MAT 1275."),
    ("Can I take CET1100 after completing ENG1101?",
     "Yes, you can take CET 1100 after completing ENG 1101 because CET 1100 requires no prerequisite."),
    ("What courses should I take after CET1111??",
     "CET 1120 (1 credit), CET 1150 (3 credits), CET 1211 (2 credits), MAT 1375 (4 credits), ENG 1121 (3 credits)."),
    ("Can I overload credits this semester?",
     " It's best to consult with your academic advisor or the registrar's office at your institution to understand the specific requirements and process for requesting a credit overload."),
    ("I just completed CET1111 and MAT1275. What courses should I register for next semester?",
     "Based on the courses you've completed (CET 1111 and MAT 1275), you can consider registering for the following courses next semester: CET 1120 (1 credit) if you haven't taken it yet, CET 1150 (3 credits), CET 1211 (2 credits), MAT 1375 (4 credits), ENG 1101 (3 credits)."),
    ("Can I take CET2305 if I haven’t finished CET1150 yet?",
     "No you can  not take CET2350 before taing CET1150 because CET1150 is prerequisite of CET1250, and CET1250 is prerequisite of CET2350. Since you havent completed CET1150, I assume you have not finished CET2350 as well, therefore you cant not take CET2350."),
    ("I registered late and some CET classes are full. What can I do?",
     "Talk to your advisor."),
    ("I failed CET2450. Can I still register for upper-level CET classes?",
     "CET2450 is not prerequired for any higher level CET classes so even  if you failed this class you can still take upper level CET classes."),
    # Medium
    ("The courses of the CET department have been changed to EMT to CET. I have failed in EMT1255, which class I should take as an equivalent class to EMT 1255.",
     "If you failed EMT 1255, you should take CET 2350, as EMT 1255 is equivalent to CET 2350."),
    ("I have completed CET1111, CET1150, and ENG1101. What courses can I take next?",
     "You can take CET 1100, CET 1120, MAT 1275, CET 1211, CET 1250 in the next semester."),
    ("Can you list all prerequisites and corequisites for CET 3615?",
     "Prerequisites of CET 3615 are MAT 1575, CET 3525, PHY 1434 or PHY 1442."),
    ("Which general education courses are required for graduation?",
     """For degree in Computer Engineering Technology (CET), the General Education requirements typically include courses in English, Mathematics, and Flex Core electives. Here are the general education courses you need to complete:
ENG 1101 (3 credits), ENG 1121 (3 credits), MAT 1275 (4 credits), Flex Core 1 (3 credits), Flex Core 2 (3 credits), Flex core 2, Flex core 4, MAT 1375 (4 credits), MAT 1475 (4 credits), PHY 1433(4 credits), MAT 2680, MAT 2580, ID Course."""
    ),
    ("How many credits can I take if I want to overload?",
     "Generally, students are allowed to take a standard full-time course load, which is often around 12-18 credits per semester. To find out the specific number of credits you can take when overloading, and the process to request an overload, you should Consult with your Academic Advisor."),
    ("Can I take an internship while I'm still completing my last two CET courses?",
     "Yes you can take an internship even if you havent completed last two CET  courses since the internship lasses has no prerequisite."),
    ("What are the General Education requirements for my AAS degree in CET?",
     "For an AAS degree in Computer Engineering Technology (CET), the general education requirements are ENG 1101 (3 credits) - English Composition 1, ENG 1121 (3 credits), MAT 1275 (4 credits), MAT 1375 (4 credits) - The next math course after MAT 1275, PHY 1433 (4 credits), Flex Core 1 (3 credits), Flex Core 2 (3 credits), ID Course."),
    ("I transferred from another college and completed Calculus I. Do I need to retake it here?",
     "Calculus I is equivalent to MAT 1475 at your current institute.  If you have completed Calculus I at another college, you do not need to retake it."),
    ("I'm interested in switching from AAS to BTech after finishing my AAS. What are the requirements?",
     "Talk to your advisor."),
    ("I need help choosing between CET3525 and CET3625 next semester. What should I consider?",
     "You can not take CET3625 before taking CET3525, so first cosider taking CET3525 then in the next semester take CET3625."),
    # Hard
    ("Given my completed courses (CET1111, CET1150, ENG1101), provide a custom-made step-by-step plan for the remaining semesters.",
     """Based on the courses you've completed (CET 1111, CET 1150, ENG 1101), here's a step-by-step plan for the remaining semesters:
You've already completed first semester.
Second Semester: CET 1120 (2 credits) - No prerequisites.
CET 1100 (2 credits) - No prerequisites.
MAT 1275 (4 credits), ENG 1121 (3 credits), CET 1250(3 credits), CET 1211
Third Semester: CET 2312 (4 credits) - Prerequisite: CET 1120; Corequisite: PHY 1433.
CET 2350 (4 credits) - Prerequisite/Corequisite: CET 1250 and MAT 1375.
CET 2370 (2 credits) - Prerequisite: CET 1250.
CET 2390 (1 credit) - Prerequisite/Corequisite: CET 2370.
PHY 1433 (4 credits) - Prerequisite: MAT 1275.
MAT 1375(4 credits)
Fourth Semester: CET 2450 (3 credits) - Prerequisite: CET 2350.
CET 2455 (2 credits) - Prerequisite: CET 2370.
CET 2461 (2 credits) - Prerequisite/Corequisite: CET 2455, MAT 1475.
Technical elective, MAT 1475 (4 credits), Flex core  1 (3 credits)
Fifth Semester: CET 3510 (4 credits) - Prerequisite/Corequisite: CET 2411 and MAT 1575.
CET 3525 (4 credits) - Prerequisite/Corequisite: MAT 1575.
MAT 1575 (4 credits) - Prerequisite: MAT 1475.
PHY 1434 (3 credits), Flex core 2
Sixth Semester: MAT 2680 (3 credits) - Prerequisite: MAT 1575.
CET 3615 (4 credits) - Prerequisites: MAT 1575, CET 3525, and PHYS 1434 or PHYS 1442.
CET 3625 (1 credit) - Prerequisite: CET 3525; Corequisite: MAT 2680.
CET 3640 (3 credits) - Prerequisites: CET 2411 and CET 3510.
FLEX CORE 3, COM 1330 (3 credits)
Seventh Semester: CET 4711 (2 credits) - Prerequisite: CET 3640 and CET 4705.
MAT 2580 (3 credits) - Prerequisite/Corequisite: MAT 1575.
CET 4705 (2 credits) - Prerequisite: CET 3625 with a grade of C or better.
CET 4773 (4 credits) - Prerequisite: CET 3510.
Technical Elective 1 (4 credits), FLEX CORE 4
Eighth Semester: Technical Elective 2 (3 credits), CET 4811 (2 credits) - Prerequisites: CET 3640, CET 4711.
CET 4805 (2 credits) - Prerequisite: CET 4705.
CET 4864 (4 credits) - Prerequisites: CET 3625, MAT 2580.
ID Course."""
    ),
    ("I want to know the courses I can take in the 2nd semester. I've completed CET1120, CET 1150 but I haven't completed all the first-semester courses yet. Can you recommend some courses?",
     "For the 2nd semester, you can take CET 1100, CET 1111, MAT 1275, CET 1250, ENG 1100, Flex Core 1."),
    ("If I want to graduate in six/seven semesters instead of eight, how should I plan my courses?",
     """1st semester: CET 1100, CET 1111, CET 1120, CET 1150, ENG 1100, MAT 1275
2nd semester: CET 1211, CET 1250, CET 2312, CET 2350, MAT 1375, PHY 1433
3rd semester: MAT 1475, PHY 1434, CET 2370, CET 2390, CET 2450, Technical Elective
4th semester: CET 2455, CET 2461, CET 3510, CET 3525, Flex Core 2, MAT 1575
5th semester: CET 3615, CET 3625, MAT 2680, CET 3640, Flex Core 3, CET 4773
6th semester: CET 4705, CET 4711, Technical Elective 1, Flex Core 4, ID, MAT 2580, COM 1330
7th semester: CET 4805, CET 4811, CET 4864, Technical Elective 2, Flex Core 1, ENG 1121"""
    ),
    # Long Answer
    ("My catalog year is 2023, but I took a break. Should I follow the new 2025 curriculum now?",
     "Yes you have to follow 2025 curriculum."),
    ("List all courses till the eighth semester.",
     """1st semester: CET 1100, CET 1111, CET 1120, CET 1150, ENG 1100, MAT 1275
2nd semester: CET 1211, MAT 1375, CET 1250, ENG 1121, PHY 1433
3rd semester: CET 2312, MAT 1475, PHY 1434, CET 2350, CET 2370, CET 2390
4th semester: CET 2450, CET 2455, CET 2461, Technical Elective, MAT 1575
5th semester: Flex Core 1, CET 3510, CET 3525, MAT 2680, ID
6th semester: Flex Core 2, CET 3615, CET 3625, CET 3640, Technical Elective 1
7th semester: CET 4705, CET 4711, CET 4773, Flex Core 3, Technical Elective 2
8th semester: CET 4811, CET 4864, CET 4805, COM 1330, Flex Core 4"""
    )
]

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two text strings"""
    if isinstance(text1, list):
        text1 = " ".join(text1)
    if isinstance(text2, list):
        text2 = " ".join(text2)

    # Create vectorizer and transform texts
    vectorizer = CountVectorizer().fit_transform([str(text1), str(text2)]) # Ensure inputs are strings
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectors)[0, 1]
    return cosine_sim

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default_session"
    if "last_input" not in st.session_state:
        st.session_state.last_input = None
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []
    if "evaluation_complete" not in st.session_state:
        st.session_state.evaluation_complete = False
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "in_evaluation_mode" not in st.session_state:
        st.session_state.in_evaluation_mode = False
    if "api_key" not in st.session_state: # Already handled above, but good for completeness
        st.session_state.api_key = st.secrets["OPENAI_API_KEY"]


def format_message(text, is_user=False):
    """Format chat messages with styling"""
    if is_user:
        align = "right"
        bg_color = "linear-gradient(135deg, #6366f1, #4f46e5)"
        border_radius = "20px 20px 5px 20px"
    else:
        align = "left"
        bg_color = "linear-gradient(135deg, #1e1e38, #242447)"
        border_radius = "20px 20px 20px 5px"

    return f"""
    <div style="display: flex; justify-content: {align}; margin: 15px 0;">
        <div style="background: {bg_color};
                    padding: 15px 20px;
                    border-radius: {border_radius};
                    max-width: 80%;
                    font-size: 16px;
                    line-height: 1.6;
                    color: white;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                    animation: fadeIn 0.5s ease-out;
                    border: 1px solid rgba(255, 255, 255, 0.1);">
            {text}
        </div>
    </div>
    """

def custom_css():
    """Define custom CSS for the app"""
    return """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stTextInput > div > div > input {
            background-color: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 15px 20px;
            font-size: 16px;
            transition: all 0.3s ease;
            color: black; /* Ensure input text is black */
        }

        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 15px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(79, 70, 229, 0.2);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(79, 70, 229, 0.3);
        }

        .main {
            background: linear-gradient(135deg, #f8fafc, #eef2ff);
        }
    </style>
    """

def handle_submit():
    """Handle user input submission"""
    if st.session_state.user_input and len(st.session_state.user_input.strip()) > 0:
        user_message = st.session_state.user_input.strip()

        # Prevent duplicate messages
        if st.session_state.last_input != user_message:
            st.session_state.last_input = user_message

            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": user_message
            })

            # Get response from the chain
            try:
                result = conversational_rag_chain.invoke(
                    {"input": user_message},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )["answer"]
            except Exception as e:
                result = f"Error communicating with the model: {str(e)}. Please check your API key and network."
                st.error(result)


            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })

        # Reset input box
        st.session_state.user_input = ""

def run_automated_evaluation():
    """Run the automated evaluation process"""
    st.session_state.in_evaluation_mode = True
    st.session_state.evaluation_results = []
    st.session_state.messages = [] # Clear messages for evaluation display

    if not st.session_state.api_key:
        st.error("OpenAI API Key is not set. Please set it in the Admin Panel.")
        st.session_state.in_evaluation_mode = False # Exit evaluation mode
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (question, expected_answer) in enumerate(questions_answers):
        progress = (i) / len(questions_answers)
        progress_bar.progress(progress)
        status_text.text(f"Processing question {i+1}/{len(questions_answers)}")

        st.session_state.session_id = f"eval_session_{i}" # Unique session for each question

        actual_answer = "Error: Could not get response." # Default error message
        try:
            actual_answer = conversational_rag_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )["answer"]
        except Exception as e:
            actual_answer = f"Error invoking RAG chain: {str(e)}"
            st.warning(f"Error processing question '{question[:50]}...': {str(e)}")


        similarity = calculate_cosine_similarity(str(actual_answer), str(expected_answer))

        st.session_state.evaluation_results.append({
            "question": question,
            "expected_answer": str(expected_answer), # Ensure it's a string
            "actual_answer": str(actual_answer),     # Ensure it's a string
            "similarity": similarity
        })

        time.sleep(0.5) # Small delay

    progress_bar.progress(1.0)
    status_text.text("Evaluation completed!")

    save_results_to_csv()
    st.session_state.evaluation_complete = True

def save_results_to_csv():
    """Save evaluation results to a CSV file"""
    if st.session_state.evaluation_results:
        df = pd.DataFrame(st.session_state.evaluation_results)
        df.to_csv("qa_evaluation_results.csv", index=False)
        return df
    return pd.DataFrame() # Return empty DataFrame if no results

def display_evaluation_results(location="main"):
    """Display the evaluation results in the app"""
    if not st.session_state.evaluation_results:
        if location == "main": # Only show this prominent message in the main area
            st.info("No evaluation results to display. Run the evaluation from the Admin Panel.")
        return

    df = pd.DataFrame(st.session_state.evaluation_results)

    avg_similarity = df['similarity'].mean() if not df.empty else 0
    st.metric("Average Cosine Similarity", f"{avg_similarity:.4f}")

    st.write("Detailed Results:")
    tab1, tab2 = st.tabs(["Detailed View", "Summary Chart"])

    with tab1:
        for idx, row in df.iterrows():
            with st.expander(f"Question {idx + 1}: {row['question'][:100]}..."):
                st.write("**Question:**")
                st.write(row['question'])

                st.write("**LLM Answer:**")
                st.write(row['actual_answer'])

                st.write("**Expected Answer:**")
                st.write(row['expected_answer'])

                st.write("**Cosine Similarity:**")
                st.write(f"{row['similarity']:.4f}")

                if row['similarity'] >= 0.8:
                    st.success(f"High similarity: {row['similarity']:.4f}")
                elif row['similarity'] >= 0.6:
                    st.warning(f"Medium similarity: {row['similarity']:.4f}")
                else:
                    st.error(f"Low similarity: {row['similarity']:.4f}")

    with tab2:
        st.write("Similarity Scores Chart")
        if not df.empty:
            chart_data = pd.DataFrame({
                'Question': [f"Q{i+1}" for i in range(len(df))],
                'Similarity': df['similarity']
            }).set_index('Question')
            st.bar_chart(chart_data)
        else:
            st.write("No data for chart.")


    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="qa_evaluation_results.csv",
            mime="text/csv",
            key=f"download_results_csv_{location}"
        )

def main():
    st.set_page_config(page_title="Academic Advisement Bot", layout="wide")
    init_session_state()

    st.markdown(custom_css(), unsafe_allow_html=True)

    st.markdown("""
        <div style="display: flex; justify-content: center; width: 100%; margin: 0 auto;">
            <div style="display: inline-block; text-align: center; padding: 4px 20px;
                        background: linear-gradient(135deg, #4f46e5, #3b82f6);
                        border-radius: 6px; margin: 4px auto;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(10px);
                        max-width: fit-content;">
                <div style="display: flex; flex-direction: column; gap: 2px;">
                    <h1 style="color: white; font-size: 18px; font-weight: 600; margin: 0; padding: 0;
                                font-family: 'Arial', sans-serif; line-height: 1.2;
                                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
                        Academic Advisement Bot
                    </h1>
                    <p style="color: rgba(255, 255, 255, 0.95); font-size: 11px; font-weight: 400;
                                margin: 0; padding: 0; line-height: 1.2;
                                letter-spacing: 0.3px;">
                        Ask questions about your course material!
                    </p>
                </div>
            </div>
        </div>
        <style>
            div[data-testid="stVerticalBlock"] > div:first-child {
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Admin Panel")

        st.subheader("API Key Management")
        masked_api_key = st.session_state.api_key[:4] + "..." + st.session_state.api_key[-4:] if st.session_state.api_key and len(st.session_state.api_key) > 8 else "Not set"
        st.text(f"Current API Key: {masked_api_key}")

        new_api_key_input = st.text_input("Set/Update API Key", type="password", key="new_api_key_input_field")
        if st.button("Update API Key"):
            if new_api_key_input:
                success, message = update_api_key(new_api_key_input)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter a new API key.")

        st.divider()

        if st.button("Run Automated Evaluation"):
            run_automated_evaluation()

        if st.session_state.evaluation_complete and st.session_state.in_evaluation_mode: # Show in sidebar only during/after eval
            st.success("Evaluation completed! Results saved to qa_evaluation_results.csv")
            display_evaluation_results(location="sidebar")


    if not st.session_state.in_evaluation_mode:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                st.markdown(
                    format_message(
                        message["content"],
                        message["role"] == "user"
                    ),
                    unsafe_allow_html=True
                )

        st.markdown("<div style='position: fixed; bottom: 0; left: 0; right: 0; padding: 20px; background: white; box-shadow: 0 -4px 15px rgba(0, 0, 0, 0.1); z-index: 999;'>", unsafe_allow_html=True)
        cols = st.columns([7, 1])
        with cols[0]:
            st.text_input(
                "Message",
                key="user_input",
                placeholder="Ask about courses...",
                on_change=handle_submit,
                label_visibility="collapsed"
            )
        with cols[1]:
            st.button("Send", on_click=handle_submit, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else: # Evaluation mode display in main area
        if st.session_state.evaluation_complete:
            st.subheader("Evaluation Results (Main View)")
            display_evaluation_results(location="main_eval_results") # Unique key for download button
            if st.button("Return to Chat Mode"):
                st.session_state.in_evaluation_mode = False
                st.session_state.messages = [] # Clear eval messages
                st.rerun()
        else:
            st.info("Evaluation in progress... Please wait. Results will appear here and in the sidebar once complete.")


    st.markdown("""
        <div style="text-align: center; padding: 20px 0; color: #6b7280; font-size: 14px; padding-bottom: 70px;"> Built with ❤️ to help students succeed
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
