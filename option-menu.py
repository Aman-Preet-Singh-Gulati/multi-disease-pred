import streamlit as st
from streamlit_option_menu import option_menu


# 1. As sidebar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title = "Main Menu", # required
#         options = ["Home", "Projects", "Contact"], # required
#         icons = ["house", "book", "envelope"],
#         menu_icon = "cast",
#         default_index = 0
#     ) 

# 2. Horizontal menu
selected = option_menu(
    menu_title = "Main Menu", # required
    options = ["Home", "Projects", "Contact"], # required
    icons = ["house", "book", "envelope"],
    menu_icon = "cast",
    default_index = 0,
    orientation ="horizontal",
) 

if selected == "Home":
    st.title(f"You have selected {selected}")
if selected == "Projects":
    st.title(f"You have selected {selected}")
if selected == "Contact":
    st.title(f"You have selected {selected}")



# st.title("Streamlit Option Menu")