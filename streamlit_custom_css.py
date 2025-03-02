custom_css = """
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e1e1e1;
    }
    
    /* Streamlit elements customization */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 16px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: none;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    /* Custom message styling */
    .semester-block {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    .semester-title {
        font-weight: bold;
        color: #4f46e5;
        margin-bottom: 15px;
        font-size: 1.2em;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 5px;
    }
    
    .course-list {
        margin: 0 0 15px 20px;
    }
    
    .course-item {
        margin-bottom: 8px;
        padding-left: 15px;
        position: relative;
    }
    
    .prerequisites {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
    }
    
    .prerequisites-title {
        font-weight: bold;
        color: #3b82f6;
        margin-bottom: 10px;
        font-size: 0.95em;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""